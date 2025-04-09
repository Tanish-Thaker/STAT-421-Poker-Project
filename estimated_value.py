def calculate_ev(win_percentage, my_stake, opponent_stake):
    """
    Calculate the expected value (EV) of a poker hand.

    Parameters:
        win_percentage (float): The probability of winning the hand (0 to 1).
        my_stake (float): The amount of money I have staked.
        opponent_stake (float): The amount of money the opponent has staked.

    Returns:
        float: The expected value of the hand.
    """
    total_pot = my_stake + opponent_stake
    ev = (win_percentage * total_pot) - ((1 - win_percentage) * my_stake)
    return ev