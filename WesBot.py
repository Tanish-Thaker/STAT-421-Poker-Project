import texasholdem as th
import texasholdem.evaluator as eval
import numpy as np
import pandas as pd
import random

import HandOddsCalcWes

############################## OBJECT DEFINITIONS ##############################

class Decision:
  def __init__(self, type:str, size:float = 0.0):
    self.type = type
    self.size = size

class PokerBot:
  def __init__(self, k:int = 10, EV_weight:float = 1.0, maturity:int = 50):
    self.k = k
    self.age = int(0)
    self.maturity = maturity
    self.EV_weight = EV_weight
    self.round_cols = {"decision":object, "size":float, "EV":float, "PHC":float,
                       "PP":float, "PTP":float, "PTK":float, "PS":float,
                       "PF":float, "PFH":float, "PFK":float, "PSF":float}
    self.hist_cols = {"decision":str, "size":float, "EV":float, "PHC":float,
                      "PP":float, "PTP":float, "PTK":float, "PS":float,
                      "PF":float, "PFH":float, "PFK":float, "PSF":float,
                      "outcome":float}
    self.round = \
      pd.DataFrame(columns = self.round_cols.keys()).astype(self.round_cols)
    self.history = \
      pd.DataFrame(columns = self.hist_cols.keys()).astype(self.hist_cols)
  
  def _log_decision_(self, decision:Decision, EV:float, hand_probs:np.ndarray):
    # append decision to `self.round`
    new_decision = pd.DataFrame({"decision":[decision.type],
                                 "size":[decision.size], "EV":[EV],
                                 "PHC":[hand_probs[0]], "PP":[hand_probs[1]],
                                 "PTP":[hand_probs[2]], "PTK":[hand_probs[3]],
                                 "PS":[hand_probs[4]], "PF":[hand_probs[5]],
                                 "PFH":[hand_probs[6]], "PFK":[hand_probs[7]],
                                 "PSF":[hand_probs[8]]})
    self.round = pd.concat([self.round, new_decision])
  
  def log_round(self, outcome:float):
    # append entire round to `self.history`
    self.round["outcome"] = [outcome] * self.round.shape[0]
    self.history = pd.concat([self.history, self.round])
    self.age += 1
  
  def _normalize_(self):
    # normalize "distance parameters" for more reliability
    hist_norm = self.history.copy()
    cols_to_norm = hist_norm.columns[2:12]
    for col in cols_to_norm:
      hist_norm[col] = \
        (hist_norm[col]-hist_norm[col].mean())/hist_norm[col].std()
    return hist_norm
  
  def _get_dist_(self, EV:float, hand_probs:np.ndarray, row:tuple):
    probs_dist = float((EV - row[2])**2)
    for i in range(hand_probs.shape):
      probs_dist += ((hand_probs[i] - row[i+3])**2)/(9*self.EV_weight)
  
  def make_decision(self, EV:float, hand_probs:np.ndarray, game:th.TexasHoldEm):
    player = game.current_player
    # determine maturity
    mature = (self.age >= self.maturity)
    if (mature):
      for decision in ["CALL/CHECK", "FOLD", "RAISE"]:
        if (np.sum(self.history["decision"] == decision) <= 2*self.k):
          mature = False
          break
    # act depending on maturity
    if (not mature):
      # too young, act randomly to get some data to work with
      babys_decision = np.random.choice(2)
      if (babys_decision == 0):
        # baby will call/check if possible
        if (game.validate_move(player, th.ActionType.CALL) or
            game.validate_move(player, th.ActionType.CHECK)):
          decision = Decision("CALL/CHECK")
        elif (game.validate_move(player, th.ActionType.ALL_IN)):
          decision = Decision("ALLIN")
        else:
          decision = Decision("FOLD")
      elif (babys_decision == 1):
        # baby will fold
        decision = Decision("FOLD")
      else:
        # baby will raise if possible
        min_raise = game.get_available_moves().raise_range[0]
        max_raise = np.min([game.players[player].chips,
                            game.get_available_moves().raise_range[-1]])
        if (min_raise <= max_raise and
            game.validate_move(player, th.ActionType.RAISE, min_raise) and
            game.validate_move(player, th.ActionType.RAISE, max_raise)):
          decision = Decision("RAISE", np.random.uniform(min_raise, max_raise))
        elif (game.validate_move(player, th.ActionType.CALL) or
              game.validate_move(player, th.ActionType.CHECK)):
          decision = Decision("CALL/CHECK")
        elif (game.validate_move(player, th.ActionType.ALL_IN)):
          decision = Decision("ALLIN")
        else:
          decision = Decision("FOLD")
    else:
      # mature, actually make decisions
      # first, split normalized history by action type
      hist_norm = self._normalize_()
      hist_call_check = \
        hist_norm.loc[hist_norm["decision"] == "CALL/CHECK"].copy()
      hist_fold = \
        hist_norm.loc[hist_norm["decision"] == "FOLD"].copy()
      hist_raise = \
        hist_norm.loc[hist_norm["decision"] == "RAISE"].copy()
      # next, get weighted distances to current game state
      dists_call_check = np.zeros(hist_call_check.shape[0], dtype = float)
      dists_fold = np.zeros(hist_fold.shape[0], dtype = float)
      dists_raise = np.zeros(hist_raise.shape[0], dtype = float)
      for i in hist_call_check.itertuples():
        dists_call_check[i] = self._get_dist_(EV, hand_probs, i)
      for i in hist_fold.itertuples():
        dists_fold[i] = self._get_dist_(EV, hand_probs, i)
      for i in hist_raise.itertuples():
        dists_raise[i] = self._get_dist_(EV, hand_probs, i)
      # now, select k closest distanced points and sum their outcomes
      nn_call_check = np.argpartition(dists_call_check, self.k)[:self.k]
      nn_fold = np.argpartition(dists_fold, self.k)[:self.k]
      nn_raise = np.argpartition(dists_raise, self.k)[:self.k]
      val_sum_call_check = np.sum(hist_call_check.loc[nn_call_check, "outcome"])
      val_sum_fold = np.sum(hist_fold.loc[nn_fold, "outcome"])
      val_sum_raise = np.sum(hist_raise.loc[nn_raise, "outcome"])
      # record and return the "best" nearest decisions
      if (val_sum_call_check > val_sum_fold and
          val_sum_call_check > val_sum_raise):
        # call/check is "best"
        # ensure decision is possible
        if (game.validate_move(player, th.ActionType.CALL) or
            game.validate_move(player, th.ActionType.CHECK)):
          decision = Decision("CALL/CHECK")
        elif (game.validate_move(player, th.ActionType.ALL_IN)):
          decision = Decision("ALLIN")
        else:
          decision = Decision("FOLD")
      elif (val_sum_fold > val_sum_raise):
        # fold is "best"
        decision = Decision("FOLD")
      else:
        # raise is "best"
        min_raise = np.max([
          game.get_available_moves().raise_range[0],  # min from game state
          np.min(hist_raise.loc[nn_raise, "size"])])  # min from agent
        max_raise = np.min([
          game.players[player].chips,                 # max from chip count
          game.get_available_moves().raise_range[-1], # max from game state
          np.max(hist_raise.loc[nn_raise, "size"])])  # max from agent
        # ensure decision is possible
        if (min_raise <= max_raise and
            game.validate_move(player, th.ActionType.RAISE, min_raise) and
            game.validate_move(player, th.ActionType.RAISE, max_raise)):
          decision = Decision("RAISE", np.random.uniform(min_raise, max_raise))
        elif (game.validate_move(player, th.ActionType.CALL) or
              game.validate_move(player, th.ActionType.CHECK)):
          decision = Decision("CALL/CHECK")
        elif (game.validate_move(player, th.ActionType.ALL_IN)):
          decision = Decision("ALLIN")
        else:
          decision = Decision("FOLD")
    # log and return decision
    self._log_decision_(decision, EV, hand_probs)
    return decision

################################## OBJECT END ##################################