
from typing import List, Tuple, Union, Optional
from enum import IntEnum


class GameOutcome(IntEnum):
    """Enumeration for game outcomes from player's perspective."""
    ONGOING = 0
    TIE = 1
    WIN = 2
    LOSS = -1


class PlayerStatus:
    """Constants for player status."""
    ALIVE = "alive"
    BUST = "bust"


class BlackjackJudger:
    """
    A comprehensive judger class for Blackjack game logic.

    This class handles all scoring and game outcome determination for Blackjack,
    including proper Ace handling and game state evaluation.

    Attributes:
        CARD_VALUES (dict): Mapping of card ranks to their base values
        BLACKJACK_TARGET (int): Target score for Blackjack (21)
        ACE_HIGH_VALUE (int): High value for Ace (11)
        ACE_LOW_VALUE (int): Low value for Ace (1)
        DEALER_STAND_THRESHOLD (int): Minimum score dealer must reach (17)
    """

    # Class constants for better maintainability
    CARD_VALUES = {
        "A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
        "9": 9, "T": 10, "J": 10, "Q": 10, "K": 10
    }

    BLACKJACK_TARGET = 21
    ACE_HIGH_VALUE = 11
    ACE_LOW_VALUE = 1
    ACE_VALUE_DIFFERENCE = ACE_HIGH_VALUE - ACE_LOW_VALUE  # 10
    DEALER_STAND_THRESHOLD = 17

    def __init__(self, np_random):
        """
        Initialize a BlackJack judger class.

        Args:
            np_random: NumPy random state for any random operations
        """
        self.np_random = np_random

    def judge_score(self, cards: List) -> int:
        """
        Calculate the optimal score for a given set of cards.

        This method implements proper Blackjack scoring rules:
        - Number cards (2-9) are worth their face value
        - Face cards (T, J, Q, K) are worth 10 points
        - Aces are worth 11 points, but can be counted as 1 to avoid busting

        Args:
            cards: A list of Card objects representing the hand

        Returns:
            The optimal score for the hand (int)

        Raises:
            ValueError: If cards list is None or contains invalid cards
            AttributeError: If card objects don't have rank attribute

        Examples:
            >>> judger = BlackjackJudger(np_random)
            >>> # Hand with Ace and King (Blackjack)
            >>> score = judger.judge_score([Card('S', 'A'), Card('H', 'K')])
            >>> assert score == 21
            >>> # Hand with two Aces (should count as 12, not 22)
            >>> score = judger.judge_score([Card('S', 'A'), Card('H', 'A')])
            >>> assert score == 12
        """
        if cards is None:
            raise ValueError("Cards cannot be None")

        if not isinstance(cards, list):
            raise ValueError("Cards must be provided as a list")

        if not cards:
            return 0

        total_score = 0
        ace_count = 0

        # Calculate base score and count Aces
        for card in cards:
            if not hasattr(card, 'rank'):
                raise AttributeError(f"Card object must have 'rank' attribute: {card}")

            rank = card.rank
            if rank not in self.CARD_VALUES:
                raise ValueError(f"Invalid card rank: {rank}")

            card_value = self.CARD_VALUES[rank]
            total_score += card_value

            if rank == 'A':
                ace_count += 1

        # Optimize Ace values to avoid busting while maximizing score
        while total_score > self.BLACKJACK_TARGET and ace_count > 0:
            total_score -= self.ACE_VALUE_DIFFERENCE
            ace_count -= 1

        return total_score

    def judge_round(self, player) -> Tuple[str, int]:
        """
        Evaluate a player's current hand status and score.

        Args:
            player: Player object with a 'hand' attribute containing cards

        Returns:
            A tuple containing:
                - status (str): PlayerStatus.ALIVE or PlayerStatus.BUST
                - score (int): The player's current optimal score

        Raises:
            AttributeError: If player doesn't have a 'hand' attribute

        Examples:
            >>> status, score = judger.judge_round(player)
            >>> if status == PlayerStatus.BUST:
            >>>     print(f"Player busted with score {score}")
        """
        if not hasattr(player, 'hand'):
            raise AttributeError("Player object must have 'hand' attribute")

        score = self.judge_score(player.hand)

        if score <= self.BLACKJACK_TARGET:
            return PlayerStatus.ALIVE, score
        else:
            return PlayerStatus.BUST, score

    def determine_game_outcome(self, player_status: str, player_score: int,
                             dealer_status: str, dealer_score: int) -> int:
        """
        Determine the game outcome for a player against the dealer.

        Game outcome rules:
        - Player bust (regardless of dealer) => LOSS (-1)
        - Dealer bust and player not bust => WIN (2)
        - Both alive: higher score wins, equal scores tie

        Args:
            player_status: Player's status (alive/bust)
            player_score: Player's score
            dealer_status: Dealer's status (alive/bust)
            dealer_score: Dealer's score

        Returns:
            GameOutcome value representing the result from player's perspective
        """
        # Player bust always loses
        if player_status == PlayerStatus.BUST:
            return GameOutcome.LOSS

        # Dealer bust with player alive means player wins
        if dealer_status == PlayerStatus.BUST:
            return GameOutcome.WIN

        # Both alive - compare scores
        if player_score > dealer_score:
            return GameOutcome.WIN
        elif player_score < dealer_score:
            return GameOutcome.LOSS
        else:
            return GameOutcome.TIE

    def judge_game(self, game, game_pointer: int) -> None:
        """
        Judge the winner of the game and update game state.

        This method maintains backward compatibility with the existing game
        implementation by directly modifying the game.winner dictionary.

        Args:
            game: Game object with players, dealer, and winner attributes
            game_pointer: Index of the current player to judge

        Raises:
            IndexError: If game_pointer is out of range
            AttributeError: If game object lacks required attributes
        """
        if not hasattr(game, 'players') or not hasattr(game, 'dealer') or not hasattr(game, 'winner'):
            raise AttributeError("Game object must have 'players', 'dealer', and 'winner' attributes")

        if game_pointer < 0 or game_pointer >= len(game.players):
            raise IndexError(f"Invalid game_pointer: {game_pointer}")

        player = game.players[game_pointer]
        dealer = game.dealer

        # Determine outcome using the new method
        outcome = self.determine_game_outcome(
            player.status, player.score,
            dealer.status, dealer.score
        )

        # Update game state (maintaining backward compatibility)
        player_key = f'player{game_pointer}'
        game.winner[player_key] = outcome

    def is_blackjack(self, cards: List) -> bool:
        """
        Check if a hand represents a natural Blackjack (21 with 2 cards).

        Args:
            cards: List of Card objects

        Returns:
            True if the hand is a natural Blackjack, False otherwise
        """
        if len(cards) != 2:
            return False

        score = self.judge_score(cards)
        return score == self.BLACKJACK_TARGET

    def should_dealer_hit(self, dealer_score: int) -> bool:
        """
        Determine if dealer should hit based on standard Blackjack rules.

        Args:
            dealer_score: Current dealer score

        Returns:
            True if dealer should hit (score < 17), False if dealer should stand
        """
        return dealer_score < self.DEALER_STAND_THRESHOLD
