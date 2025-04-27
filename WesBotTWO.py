import texasholdem as th
import texasholdem.evaluator as eval
import numpy as np
import pandas as pd
import random

import HandOddsCalcWes as hoc

############################## OBJECT DEFINITIONS ##############################

class Decision:
  def __init__(self, type:str, size:int = 0):
    self.type = type
    self.size = size
  
  def copy(self):
    output = Decision(self.type, self.size)
    return output

class PokerBot:
  def __init__(self, maturity:int = 50):
    self.age = int(0)
    self.maturity = maturity
    self.bound_top = float(0.0) # the first bound (inclusive)
    self.bound_bot = float(0.0) # the second bound (inclusive)
    self.round_cols = {"decision":str, "size":int, "EV":float}
    self.hist_cols = {"decision":str, "size":int, "EV":float,
                      "outcome":int}
    self.round = \
      pd.DataFrame(columns = self.round_cols.keys()).astype(self.round_cols)
    self.history = \
      pd.DataFrame(columns = self.hist_cols.keys()).astype(self.hist_cols)
  
  def _log_decision_(self, decision:Decision, EV:float):
    # append decision to `self.round`
    new_decision = pd.DataFrame({"decision":[decision.type],
                                 "size":[decision.size], "EV":[EV]})
    self.round = pd.concat([self.round, new_decision])
    return None
  
  def log_round(self, outcome:float):
    # append entire round to `self.history`
    self.round["outcome"] = [outcome] * self.round.shape[0]
    self.history = pd.concat([self.history, self.round])
    self.round = \
      pd.DataFrame(columns = self.round_cols.keys()).astype(self.round_cols)
    self.age += 1
    return None
  
  def _update_bounds_(self):
    # adjust upper bound
    raise_hist = self.history.loc\
      [self.history["decision"] == "RAISE"].sort_values(by = "EV")
    rhs_sums = np.zeros(raise_hist.shape[0])
    for i in range(rhs_sums.shape[0]):
      rhs_sums[i] = np.sum(raise_hist["outcome"][i:raise_hist.shape[0]])
    # set top bound as EV that maximizes sum of outcomes of big EVs (inclusive)
    self.bound_top = raise_hist.iloc[int(np.argmax(rhs_sums)), 2]
    # adjust lower bound by same method, this time for call/check
    cc_hist = self.history.loc\
      [self.history["decision"] == "CALL/CHECK"].sort_values(by = "EV")
    rhs_sums = np.zeros(cc_hist.shape[0])
    for i in range(rhs_sums.shape[0]):
      rhs_sums[i] = np.sum(cc_hist["outcome"][i:cc_hist.shape[0]])
    # set top bound as EV that maximizes sum of outcomes of big EVs (inclusive)
    self.bound_bot = cc_hist.iloc[int(np.argmax(rhs_sums)), 2]
    return None
  
  def make_decision(self, EV:float, game:th.TexasHoldEm):
    # branch on baby status
    if (self.age < self.maturity):
      # baby does the decision
      if (np.random.random() <= 0.5):
        # baby gonna call/check
        decision = Decision("CALL/CHECK")
      else:
        # baby gonna raise
        min_raise = int(game.get_available_moves().raise_range.start)
        max_raise = int(np.min([game.players[0].chips,
                                game.get_available_moves().raise_range.stop]))
        # make sure we only raise by some reasonable amount
        max_raise = int((3*min_raise + max_raise) // 4)
        if (max_raise - min_raise > 20):
          max_raise = min_raise + 20
        decision = \
          Decision("RAISE", int(np.random.uniform(min_raise, max_raise)))
    else:
      # Jerry actually does some goddamn work
      self._update_bounds_()
      if (EV < self.bound_bot):
        # smol EV, wanna fold
        decision = Decision("FOLD")
      elif (EV < self.bound_top):
        # ok EV, wanna call/check
        decision = Decision("CALL/CHECK")
      else:
        # big EV, wanna raise
        min_raise = int(game.get_available_moves().raise_range.start)
        max_raise = int(np.min([game.players[0].chips,
                                game.get_available_moves().raise_range.stop]))
        # make sure we only raise by some reasonable amount
        max_raise = int((3*min_raise + max_raise) // 4)
        if (max_raise - min_raise > 20):
          max_raise = min_raise + 20
        decision = \
          Decision("RAISE", int(np.random.uniform(min_raise, max_raise)))
        if (min_raise >= max_raise):
          decision = Decision("ALLIN")
    # now actually validate and make the decision
    if (decision.type == "RAISE"):
      if (not game.validate_move(action = th.ActionType.RAISE, value = min_raise)):
        # wanna raise, but can't, try to call/check, fold as last resort
        if (game.validate_move(action = th.ActionType.CALL) or
            game.validate_move(action = th.ActionType.CHECK)):
          decision = Decision("CALL/CHECK")
        else:
          decision = Decision("FOLD")
    elif (decision.type == "CALL/CHECK"):
      if (not (game.validate_move(action = th.ActionType.CALL) or
            game.validate_move(action = th.ActionType.CHECK))):
        # wanna call/check but can't, so fold
        decision = Decision("FOLD")
    # log and return the decision
    self._log_decision_(decision, EV)
    return decision

################################## OBJECT END ##################################