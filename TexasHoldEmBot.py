import texasholdem as th
import torch

############################### HELPER FUNCTIONS ###############################

# get_features(game:th.TexasHoldem)
#   extracts features from board state
# INPUTS:
#   game: a TexasHoldEm object containing the entire board state
# OUTPUT:
#   1 x p tensor of data features for use in making betting decisions
# OUTPUT STRUCTURE:
#   0 - 3     : one-hot encoded suit of first card
#   4 - 16    : one-hot encoded rank of first card
#   17 - 20   : one-hot encoded suit of second card
#   21 - 33   : one-hot encoded rank of second card
#   34        : do I have the button?
#   35        : do I have big blind?
#   36        : do I have small blind?
#   37        : number of my chips at stake
#   38        : number of opponent's chips at stake
#   39        : number of chips I need to call
#   40        : size of last raise
#   41 - 45   : one-hot encoded opponent's last action
#   46 - 49   : one-hot encoded suit of first community card (if exists)
#   50 - 62   : one-hot encoded rank of first community card (if exists)
#   63 - 66   : one-hot encoded suit of second community card (if exists)
#   67 - 79   : one-hot encoded rank of second community card (if exists)
#   80 - 83   : one-hot encoded suit of third community card (if exists)
#   84 - 96   : one-hot encoded rank of third community card (if exists)
#   97 - 100  : one-hot encoded suit of fourth community card (if exists)
#   101 - 113 : one-hot encoded rank of fourth community card (if exists)
#   114 - 117 : one-hot encoded suit of fifth community card (if exists)
#   118 - 130 : one-hot encoded rank of fifth community card (if exists)
def get_features(game:th.TexasHoldEm):

  # will be output with all features
  features = torch.zeros([1, 131])

  # card suit conversions
  suit_to_int = {1:0, 2:1, 4:2, 8:3}

  # hand phase name conversions
  hand_phase_to_num_cards = {'PREFLOP':0, 'FLOP':3, 'TURN':4, 'RIVER':5}

  # get all cards
  my_hand = game.get_hand(game.current_player)
  suit0, rank0 = suit_to_int[my_hand[0].suit], my_hand[0].rank
  suit1, rank1 = suit_to_int[my_hand[1].suit], my_hand[1].rank

  # one-hot encode card information
  features[0, suit0] = 1
  features[0, 4 + rank0] = 1
  features[0, 17 + suit1] = 1
  features[0, 21 + rank1] = 1

  # encode if I have button, big blind, and small blind
  features[0, 34] = int(game.btn_loc == game.current_player)
  features[0, 35] = int(game.bb_loc == game.current_player)
  features[0, 36] = int(game.sb_loc == game.current_player)

  # encode pot information
  features[0, 37] = game.chips_at_stake(game.current_player)
  features[0, 38] = game.chips_at_stake(int(not bool(game.current_player)))
  features[0, 39] = game.chips_to_call(game.current_player)
  features[0, 40] = game.last_raise

  # encode last player's action, if it exists
  if (game.action != (None, None)):
    features[0, 41 + (game.action[0].value - 1)] = 1 # the enum starts at 1
  
  # encode community cards
  num_community_cards = hand_phase_to_num_cards[game.hand_phase.name]
  if (num_community_cards >= 3):
    suit2, rank2 = suit_to_int[game.board[0].suit], game.board[0].rank
    suit3, rank3 = suit_to_int[game.board[1].suit], game.board[1].rank
    suit4, rank4 = suit_to_int[game.board[2].suit], game.board[2].rank
    features[0, 46 + suit2] = 1
    features[0, 50 + rank2] = 1
    features[0, 63 + suit3] = 1
    features[0, 67 + rank3] = 1
    features[0, 80 + suit4] = 1
    features[0, 84 + rank4] = 1
  if (num_community_cards >= 4):
    suit5, rank5 = suit_to_int[game.board[3].suit], game.board[3].rank
    features[0, 97 + suit5] = 1
    features[0, 101 + rank5] = 1
  if (num_community_cards == 5):
    suit6, rank6 = suit_to_int[game.board[4].suit], game.board[4].rank
    features[0, 114 + suit6] = 1
    features[0, 118 + rank6] = 1
  
  features.requires_grad_()

  return features

################################## GAME START ##################################

game = th.TexasHoldEm(buyin=500, big_blind=5, small_blind=2, max_players=2)
game.start_hand()
