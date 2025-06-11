import unittest
import numpy as np
from unittest.mock import Mock

from rlcard.games.blackjack.judger import BlackjackJudger, GameOutcome, PlayerStatus
from rlcard.games.base import Card


class TestBlackjackJudger(unittest.TestCase):
    """Comprehensive tests for the refactored BlackjackJudger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.np_random = np.random.RandomState(42)
        self.judger = BlackjackJudger(self.np_random)

    def test_initialization(self):
        """Test judger initialization."""
        self.assertIsNotNone(self.judger.np_random)
        self.assertEqual(self.judger.BLACKJACK_TARGET, 21)
        self.assertEqual(self.judger.DEALER_STAND_THRESHOLD, 17)

    def test_judge_score_basic_cards(self):
        """Test scoring with basic number and face cards."""
        # Test number cards
        cards = [Card('S', '2'), Card('H', '5')]
        self.assertEqual(self.judger.judge_score(cards), 7)
        
        # Test face cards
        cards = [Card('S', 'K'), Card('H', 'Q')]
        self.assertEqual(self.judger.judge_score(cards), 20)
        
        # Test ten card
        cards = [Card('S', 'T'), Card('H', '9')]
        self.assertEqual(self.judger.judge_score(cards), 19)

    def test_judge_score_with_aces(self):
        """Test scoring with Aces in various scenarios."""
        # Ace + King = 21 (Blackjack)
        cards = [Card('S', 'A'), Card('H', 'K')]
        self.assertEqual(self.judger.judge_score(cards), 21)
        
        # Ace + 6 = 17 (Ace as 11)
        cards = [Card('S', 'A'), Card('H', '6')]
        self.assertEqual(self.judger.judge_score(cards), 17)
        
        # Ace + 6 + 5 = 12 (Ace as 1)
        cards = [Card('S', 'A'), Card('H', '6'), Card('D', '5')]
        self.assertEqual(self.judger.judge_score(cards), 12)
        
        # Two Aces = 12 (one as 11, one as 1)
        cards = [Card('S', 'A'), Card('H', 'A')]
        self.assertEqual(self.judger.judge_score(cards), 12)
        
        # Three Aces = 13 (one as 11, two as 1)
        cards = [Card('S', 'A'), Card('H', 'A'), Card('D', 'A')]
        self.assertEqual(self.judger.judge_score(cards), 13)
        
        # Four Aces = 14 (one as 11, three as 1)
        cards = [Card('S', 'A'), Card('H', 'A'), Card('D', 'A'), Card('C', 'A')]
        self.assertEqual(self.judger.judge_score(cards), 14)

    def test_judge_score_edge_cases(self):
        """Test edge cases for score calculation."""
        # Empty hand
        self.assertEqual(self.judger.judge_score([]), 0)
        
        # Single card
        cards = [Card('S', '7')]
        self.assertEqual(self.judger.judge_score(cards), 7)
        
        # Bust scenario
        cards = [Card('S', 'K'), Card('H', 'Q'), Card('D', '5')]
        self.assertEqual(self.judger.judge_score(cards), 25)

    def test_judge_score_error_handling(self):
        """Test error handling in score calculation."""
        # Test with None
        with self.assertRaises(ValueError):
            self.judger.judge_score(None)
        
        # Test with non-list
        with self.assertRaises(ValueError):
            self.judger.judge_score("not a list")
        
        # Test with invalid card (missing rank)
        invalid_card = Mock()
        del invalid_card.rank  # Remove rank attribute
        with self.assertRaises(AttributeError):
            self.judger.judge_score([invalid_card])
        
        # Test with invalid rank
        invalid_card = Mock()
        invalid_card.rank = 'X'  # Invalid rank
        with self.assertRaises(ValueError):
            self.judger.judge_score([invalid_card])

    def test_judge_round_alive(self):
        """Test round judging for alive players."""
        player = Mock()
        player.hand = [Card('S', '7'), Card('H', '8')]
        
        status, score = self.judger.judge_round(player)
        self.assertEqual(status, PlayerStatus.ALIVE)
        self.assertEqual(score, 15)

    def test_judge_round_bust(self):
        """Test round judging for busted players."""
        player = Mock()
        player.hand = [Card('S', 'K'), Card('H', 'Q'), Card('D', '5')]
        
        status, score = self.judger.judge_round(player)
        self.assertEqual(status, PlayerStatus.BUST)
        self.assertEqual(score, 25)

    def test_judge_round_blackjack(self):
        """Test round judging for Blackjack."""
        player = Mock()
        player.hand = [Card('S', 'A'), Card('H', 'K')]
        
        status, score = self.judger.judge_round(player)
        self.assertEqual(status, PlayerStatus.ALIVE)
        self.assertEqual(score, 21)

    def test_judge_round_error_handling(self):
        """Test error handling in round judging."""
        # Player without hand attribute
        player = Mock()
        del player.hand
        with self.assertRaises(AttributeError):
            self.judger.judge_round(player)

    def test_determine_game_outcome(self):
        """Test game outcome determination."""
        # Player bust
        outcome = self.judger.determine_game_outcome(
            PlayerStatus.BUST, 25, PlayerStatus.ALIVE, 20
        )
        self.assertEqual(outcome, GameOutcome.LOSS)
        
        # Dealer bust, player alive
        outcome = self.judger.determine_game_outcome(
            PlayerStatus.ALIVE, 20, PlayerStatus.BUST, 25
        )
        self.assertEqual(outcome, GameOutcome.WIN)
        
        # Player wins with higher score
        outcome = self.judger.determine_game_outcome(
            PlayerStatus.ALIVE, 20, PlayerStatus.ALIVE, 18
        )
        self.assertEqual(outcome, GameOutcome.WIN)
        
        # Dealer wins with higher score
        outcome = self.judger.determine_game_outcome(
            PlayerStatus.ALIVE, 18, PlayerStatus.ALIVE, 20
        )
        self.assertEqual(outcome, GameOutcome.LOSS)
        
        # Tie
        outcome = self.judger.determine_game_outcome(
            PlayerStatus.ALIVE, 20, PlayerStatus.ALIVE, 20
        )
        self.assertEqual(outcome, GameOutcome.TIE)

    def test_is_blackjack(self):
        """Test Blackjack detection."""
        # True Blackjack
        cards = [Card('S', 'A'), Card('H', 'K')]
        self.assertTrue(self.judger.is_blackjack(cards))
        
        # 21 but not Blackjack (more than 2 cards)
        cards = [Card('S', '7'), Card('H', '7'), Card('D', '7')]
        self.assertFalse(self.judger.is_blackjack(cards))
        
        # 2 cards but not 21
        cards = [Card('S', 'K'), Card('H', '9')]
        self.assertFalse(self.judger.is_blackjack(cards))
        
        # Single card
        cards = [Card('S', 'A')]
        self.assertFalse(self.judger.is_blackjack(cards))

    def test_should_dealer_hit(self):
        """Test dealer hit/stand logic."""
        # Should hit
        self.assertTrue(self.judger.should_dealer_hit(16))
        self.assertTrue(self.judger.should_dealer_hit(10))
        
        # Should stand
        self.assertFalse(self.judger.should_dealer_hit(17))
        self.assertFalse(self.judger.should_dealer_hit(20))
        self.assertFalse(self.judger.should_dealer_hit(21))

    def test_judge_game_integration(self):
        """Test the complete game judging functionality."""
        # Create mock game object
        game = Mock()
        game.players = [Mock(), Mock()]
        game.dealer = Mock()
        game.winner = {}
        
        # Set up player and dealer states
        game.players[0].status = PlayerStatus.ALIVE
        game.players[0].score = 20
        game.dealer.status = PlayerStatus.ALIVE
        game.dealer.score = 18
        
        # Judge the game
        self.judger.judge_game(game, 0)
        
        # Check result
        self.assertEqual(game.winner['player0'], GameOutcome.WIN)

    def test_judge_game_error_handling(self):
        """Test error handling in game judging."""
        # Game without required attributes
        game = Mock()
        del game.players
        with self.assertRaises(AttributeError):
            self.judger.judge_game(game, 0)
        
        # Invalid game pointer
        game = Mock()
        game.players = [Mock()]
        game.dealer = Mock()
        game.winner = {}
        with self.assertRaises(IndexError):
            self.judger.judge_game(game, 1)


if __name__ == '__main__':
    unittest.main()
