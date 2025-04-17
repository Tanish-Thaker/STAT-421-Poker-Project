import texasholdem as th
import texasholdem.evaluator as eval
import numpy as np
import random

############################### HELPER FUNCTIONS ###############################

# card_to_num(card:th.Card)
# converts th.Card object to a number from 0-51
# CARD DICTIONARIES:
#   suit = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
#   rank = {'2': 0, '3': 1, '4': 2, '5': 3,
#           '6': 4, '7': 5, '8': 6, '9': 7,
#           'T': 8, 'J': 9, 'Q': 10,
#           'K': 11, 'A': 12}
# FORMULA:
#   output = (13 * suit) + rank
def card_to_num(card:th.Card):
  suit_dict = {1:0, 2:1, 4:2, 8:3}
  return (13 * suit_dict[card.suit]) + card.rank

# num_to_card(num:int)
# converts a number from 0-51 to th.Card object
# CARD DICTIONARIES:
#   suit = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
#   rank = {0: '2', 1: '3', 2: '4', 3: '5',
#           4: '6', 5: '7', 6: '8', 7: '9',
#           8: 'T', 9: 'J', 10: 'Q',
#           11: 'K', 12: 'A'}
# FORMULA:
#   card.suit = num / 13 passed into dict
#   card.rank = num % 13 passed into dict
def num_to_card(num:int):
  suit_dict = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
  rank_dict = {0: '2', 1: '3', 2: '4', 3: '5',
               4: '6', 5: '7', 6: '8', 7: '9',
               8: 'T', 9: 'J', 10: 'Q',
               11: 'K', 12: 'A'}
  return th.Card(rank_dict[num % 13]+suit_dict[num // 13])

# get_winner(pockets:list, com_cards:list)
# returns index of hand winner (i.e. if pockets[i] has winning hand, returns i)
# INPUTS:
#   pockets:   list of lists of th.Card representing every player's pocket
#   com_cards: list of th.Card objects representing the community cards
# OUTPUT:
#   integer index of hand winner
def get_winner(pockets:list, com_cards:list):
  num_players = len(pockets)
  hand_ranks = [0] * num_players # the smallest hand rank wins the hand
  for i in range(num_players):
    hand_ranks[i] = eval.evaluate(pockets[i], com_cards)
  return hand_ranks.index(min(hand_ranks))

# get_hand_type(pocket:list, com_cards:list)
# returns hand type (e.g. Pair, Flush, etc.)
# INPUTS:
#   pocket:    list of th.Card objects representing player's pocket
#   com_cards: list of th.Card objects representing the community cards
# OUPUT:
#   integer corresponding to a hand type, according to the following dictionary
#     {"High card":0, "Pair":1, "Two Pair":2, "Three of a Kind":3, "Straight":4,
#      "Flush":5, "Full House":6, "Four of a Kind":7, "Straight Flush":8}
def get_hand_type(pocket:list, com_cards:list):
  return 9 - eval.get_rank_class(eval.evaluate(pocket, com_cards))

# estimate_win_prob(game:th.TexasHoldEm, player:int, num_players:int,
#                   num_bootstraps:int)
# from player's hand, community cards, number of players who haven't folded,
# uses bootstrap methodology to estimate the probability of player winning the
# hand by generating random opponent hands and filling in community cards
# we do not actually see the hands of other players during this calculation
# INPUTS:
#   game: a TexasHoldem object for the current game
#   player: integer of the index of player that we're evaluating
#   num_players: number of players who haven't folded (includes player)
#   num_bootstraps: number of boostraps to perform
# OUTPUT:
#   a numpy float of the probability of winning the hand
def estimate_win_prob(game:th.TexasHoldEm, player:int, num_players:int,
                      num_bootstraps:int):
  
  # bootstrap loop preparation
  wins_and_losses = np.zeros(num_bootstraps, dtype = int) # win = 1, loss = 0
  all_cards = set(range(52))
  known_cards = set() # holds int representations of cards that are known
  my_hand = game.get_hand(player)
  for card in my_hand: # populate known_cards with player's hand
    known_cards.add(card_to_num(card))
  for card in game.board: # populate known_cards with real community cards
    known_cards.add(card_to_num(card))
  unknown_cards = all_cards.difference(known_cards) # cards from here are pulled

  # bootstrap loop
  for i in range(num_bootstraps):

    pullable_cards = unknown_cards # int representation set of pullable cards
    pockets = [] # list of player hands (list of lists)
    com_cards = [] # list of community cards
    for card in game.board: # populate existing community cards
      com_cards.append(card)
    while len(com_cards) < 5: # populate remaining community cards randomly
      new_card_as_num = random.choice(list(pullable_cards))
      pullable_cards = pullable_cards.difference({new_card_as_num})
      com_cards.append(num_to_card(new_card_as_num))
    
    # populate remaining pockets randomly
    for j in range(num_players):
      if j==player:
        pockets.append(my_hand)
      else:
        new_card_as_num = random.choice(list(pullable_cards))
        pullable_cards = pullable_cards.difference({new_card_as_num})
        newer_card_as_num = random.choice(list(pullable_cards))
        pullable_cards = pullable_cards.difference({newer_card_as_num})
        pockets.append([num_to_card(new_card_as_num),
                        num_to_card(newer_card_as_num)])
    
    # determine winner of this bootstrapped hand
    wins_and_losses[i] = int(get_winner(pockets, com_cards) == player)
  
  return np.mean(wins_and_losses)

# estimate_hand_probs(game:th.TexasHoldEm, player:int, num_bootstraps:int)
# from player's hand and community cards, uses bootstrap methodology to estimate
# the probability of player getting all hand types (e.g. two pair, flush, etc)
# from the remaining unknown community cards
# INPUTS:
#   game: a TexasHoldem object for the current game
#   player: integer of the index of player that we're evaluating
#   num_bootstraps: number of boostraps to perform
# OUTPUT:
#   a numpy array of the following structure:
#     [0]: Probability of High Card
#     [1]: Probability of One Pair
#     [2]: Probability of Two Pairs
#     [3]: Probability of Three of a Kind
#     [4]: Probability of Straight
#     [5]: Probability of Flush
#     [6]: Probability of Full House
#     [7]: Probability of Four of a Kind
#     [8]: Probability of Straight Flush
def estimate_hand_probs(game:th.TexasHoldEm, player:int, num_bootstraps:int):

  # I'll only make this if needbe. For now, I'm just gonna focus on the
  # combined functionality with estimate_win_prob() to save my time.

  pass

# estimate_win_and_hand_probs(game:th.TexasHoldEm, player:int,
#                             num_players:int, num_bootstraps:int)
# from player's hand, community cards, number of players who haven't folded,
# uses bootstrap methodology to estimate the probability of player winning the
# hand and the probabilities of getting specific hands by generating random
# opponent hands and filling in community cards
# we do not actually see the hands of other players during this calculation
# INPUTS:
#   game: a TexasHoldem object for the current game
#   player: integer of the index of player that we're evaluating
#   num_players: number of players who haven't folded (includes player)
#   num_bootstraps: number of boostraps to perform
# OUTPUTS:
#   a numpy float of the probability of winning the hand
#   a numpy array of the following structure:
#     [0]: Probability of High Card
#     [1]: Probability of One Pair
#     [2]: Probability of Two Pairs
#     [3]: Probability of Three of a Kind
#     [4]: Probability of Straight
#     [5]: Probability of Flush
#     [6]: Probability of Full House
#     [7]: Probability of Four of a Kind
#     [8]: Probability of Straight Flush
def estimate_win_and_hand_probs(game:th.TexasHoldEm, player:int,
                                num_players:int, num_bootstraps:int):
  
  # bootstrap loop preparation
  wins_and_losses = np.zeros(num_bootstraps, dtype = int) # win = 1, loss = 0
  hand_types = np.zeros(num_bootstraps, dtype = int) # see get_hand_type()
  all_cards = set(range(52))
  known_cards = set() # holds int representations of cards that are known
  my_hand = game.get_hand(player)
  for card in my_hand: # populate known_cards with player's hand
    known_cards.add(card_to_num(card))
  for card in game.board: # populate known_cards with real community cards
    known_cards.add(card_to_num(card))
  unknown_cards = all_cards.difference(known_cards) # cards from here are pulled

  # bootstrap loop
  for i in range(num_bootstraps):

    pullable_cards = unknown_cards # int representation set of pullable cards
    pockets = [] # list of player hands (list of lists)
    com_cards = [] # list of community cards
    for card in game.board: # populate existing community cards
      com_cards.append(card)
    while len(com_cards) < 5: # populate remaining community cards randomly
      new_card_as_num = random.choice(list(pullable_cards))
      pullable_cards = pullable_cards.difference({new_card_as_num})
      com_cards.append(num_to_card(new_card_as_num))
    
    # populate remaining pockets randomly
    for j in range(num_players):
      if j==player:
        pockets.append(my_hand)
      else:
        new_card_as_num = random.choice(list(pullable_cards))
        pullable_cards = pullable_cards.difference({new_card_as_num})
        newer_card_as_num = random.choice(list(pullable_cards))
        pullable_cards = pullable_cards.difference({newer_card_as_num})
        pockets.append([num_to_card(new_card_as_num),
                        num_to_card(newer_card_as_num)])
    
    # determine winner of this bootstrapped hand
    wins_and_losses[i] = int(get_winner(pockets, com_cards) == player)

    # determine hand type
    hand_types[i] = get_hand_type(my_hand, com_cards)
  
  # get hand type probabilities
  hand_type_probs = np.zeros(9)
  for i in range(9):
    hand_type_probs[i] = np.mean(hand_types == i)
  
  return np.mean(wins_and_losses), hand_type_probs

################################## GAME START ##################################

# this is for you guys to see that this works, because it does :D

game = th.TexasHoldEm(buyin=500, big_blind=5, small_blind=2, max_players=2)
game.start_hand()

print("Each player's hands")
print("P1:", game.get_hand(0))
print("P2:", game.get_hand(1))
print()

game.take_action(th.ActionType.CALL)
game.take_action(th.ActionType.CHECK)

print("Flop:")
print(game.board)
print()

print("Hand evaluations")
print("P1:", eval.rank_to_string(eval.evaluate(game.get_hand(0), game.board)))
print("P2:", eval.rank_to_string(eval.evaluate(game.get_hand(1), game.board)))
print()

print("Bootstrap probabilities of winning hand (NO KNOWLEDGE OF OTHER PLAYER):")
print("P1:", estimate_win_prob(game, 0, 2, 1000))
print("P2:", estimate_win_prob(game, 1, 2, 1000))
print()

game.take_action(th.ActionType.CHECK)
game.take_action(th.ActionType.CHECK)

print("Flop and Turn:")
print(game.board)
print()

print("Hand evaluations")
print("P1:", eval.rank_to_string(eval.evaluate(game.get_hand(0), game.board)))
print("P2:", eval.rank_to_string(eval.evaluate(game.get_hand(1), game.board)))
print()

print("Bootstrap probabilities of winning hand (NO KNOWLEDGE OF OTHER PLAYER):")
print("P1:", estimate_win_prob(game, 0, 2, 1000))
print("P2:", estimate_win_prob(game, 1, 2, 1000))
print()

game.take_action(th.ActionType.CHECK)
game.take_action(th.ActionType.CHECK)

print("Flop and Turn and River:")
print(game.board)
print()

print("Hand evaluations")
print("P1:", eval.rank_to_string(eval.evaluate(game.get_hand(0), game.board)))
print("P2:", eval.rank_to_string(eval.evaluate(game.get_hand(1), game.board)))
print()

print("Bootstrap probabilities of winning hand (NO KNOWLEDGE OF OTHER PLAYER):")
print("P1:", estimate_win_prob(game, 0, 2, 1000))
print("P2:", estimate_win_prob(game, 1, 2, 1000))
print()

print("Hand ranks (LOWEST WINS THE HAND):")
print("P1:", eval.evaluate(game.get_hand(0), game.board))
print("P2:", eval.evaluate(game.get_hand(1), game.board))
