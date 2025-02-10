import pygame
import chess
import chess.svg
import io
import time

WIDTH = 800
SQ_SIZE = WIDTH // 8
TIMEOUT = 90

# Piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 1000,
    chess.KING: 20000
}

# Piece position tables
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, -20, -20, 10, 10,  5,
    5, -5, -10,  0,  0, -10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    60, 60, 60, 60, 60, 60, 60, 60,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,  0,  0,  0,  0, -20, -40,
    -30,  0, 10, 15, 15, 10,  0, -30,
    -30,  5, 15, 20, 20, 15,  5, -30,
    -30,  0, 15, 20, 20, 15,  0, -30,
    -30,  5, 10, 15, 15, 10,  5, -30,
    -40, -20,  0,  5,  5,  0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5, 10, 10,  5,  0, -10,
    -10,  5,  5, 10, 10,  5,  5, -10,
    -10,  0, 10, 10, 10, 10,  0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10,  5,  0,  0,  0,  0,  5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20, -10, -10, -5, -10, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5,  5,  5,  5,  0, -10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0, -10,
    -10,  0,  5,  0,  0,  0,  0, -10,
    -20, -10, -10, -5, -10, -10, -10, -20
]

KING_TABLE = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE
}


class ChessGUI:
    def __init__(self):
        self.board = chess.Board()
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, WIDTH), pygame.SCALED)
        self.piece_cache = {}
        self.font = pygame.font.Font(None, 36)
        self.started = False
        self.white_time = TIMEOUT
        self.black_time = TIMEOUT
        self.timer_start = None
        self.last_time_check = None
        self.current_player = chess.WHITE
        self.is_ai_turn = False if self.current_player == chess.WHITE else True
        self.delay_start_time = None
        self.promotion_choice = None
        self.last_move = None
        self.last_move_from = None  # record the starting position of the last move
        self.castling_rights = {chess.WHITE: True, chess.BLACK: True}
        self.king_moved = {chess.WHITE: False, chess.BLACK: False}
        self.rook_moved = {chess.WHITE: {chess.A1: False, chess.H1: False},
                          chess.BLACK: {chess.A8: False, chess.H8: False}}

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = (238, 238, 210) if (row + col) % 2 == 0 else (118, 150, 86)
                pygame.draw.rect(self.screen, color,
                                 (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        # Draw markers for the start and end positions of the last move
        if self.last_move_from is not None and self.last_move is not None:
            # Draw starting position
            row = 7 - (self.last_move_from // 8)
            col = self.last_move_from % 8
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill((0, 0, 255, 64))  # Use semi-transparent blue marker
            self.screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

            # Draw ending position
            to_square = self.last_move.to_square
            row = 7 - (to_square // 8)
            col = to_square % 8
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill((0, 0, 255, 64))  # Use semi-transparent blue marker
            self.screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

    def get_piece_image(self, piece):
        key = f"{piece.color}_{piece.piece_type}"
        if key not in self.piece_cache:
            svg = chess.svg.piece(piece, size=SQ_SIZE * 1.2)
            img = pygame.image.load(io.BytesIO(svg.encode('utf-8')))
            self.piece_cache[key] = pygame.transform.smoothscale(img, (SQ_SIZE, SQ_SIZE))
        return self.piece_cache[key]

    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                img = self.get_piece_image(piece)
                self.screen.blit(img, (col * SQ_SIZE, row * SQ_SIZE))

    def draw_valid_moves(self, valid_moves):
        highlight_color = (255, 255, 0, 128)
        capture_color = (255, 0, 0, 64)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)

        for move in valid_moves:
            target_square = move.to_square
            row = 7 - (target_square // 8)
            col = target_square % 8

            if self.board.piece_at(target_square) and self.board.piece_at(target_square).color != self.board.turn:
                s.fill(capture_color)
            else:
                s.fill(highlight_color)

            self.screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

    def check_checkmate(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            loser = "Black" if self.board.turn == chess.BLACK else "White"
            self.show_end_screen(f"{loser} loses! {winner} wins!")
            return True

        # Check for draw conditions
        pieces = self.board.piece_map()
        if len(pieces) <= 4:  # Only check draw conditions if the number of pieces is less than or equal to 4
            # Only kings left for both sides
            if len(pieces) == 2:
                kings_only = True
                for piece in pieces.values():
                    if piece.piece_type != chess.KING:
                        kings_only = False
                        break
                if kings_only:
                    self.show_end_screen("Draw! Only kings left.")
                    return True

            # One side has only king, the other side has king and single knight
            if len(pieces) == 3:
                white_pieces = {sq: p for sq, p in pieces.items() if p.color == chess.WHITE}
                black_pieces = {sq: p for sq, p in pieces.items() if p.color == chess.BLACK}
                if (len(white_pieces) == 1 and list(white_pieces.values())[0].piece_type == chess.KING and
                    len(black_pieces) == 2 and any(p.piece_type == chess.KNIGHT for p in black_pieces.values())) or\
                   (len(black_pieces) == 1 and list(black_pieces.values())[0].piece_type == chess.KING and
                    len(white_pieces) == 2 and any(p.piece_type == chess.KNIGHT for p in white_pieces.values())):
                    self.show_end_screen("Draw! King vs King and Knight.")
                    return True

                # One side has only king, the other side has king and single bishop
                if (len(white_pieces) == 1 and list(white_pieces.values())[0].piece_type == chess.KING and
                    len(black_pieces) == 2 and any(p.piece_type == chess.BISHOP for p in black_pieces.values())) or\
                   (len(black_pieces) == 1 and list(black_pieces.values())[0].piece_type == chess.KING and
                    len(white_pieces) == 2 and any(p.piece_type == chess.BISHOP for p in white_pieces.values())):
                    self.show_end_screen("Draw! King vs King and Bishop.")
                    return True

            # Both sides have a bishop of the same color complex
            if len(pieces) == 4:
                bishops = []
                for piece in pieces.values():
                    if piece.piece_type == chess.BISHOP:
                        bishops.append(piece)
                if len(bishops) == 2 and bishops[0].color != bishops[1].color:
                    # Check if two bishops are on same color squares
                    bishop_squares = [sq for sq, p in pieces.items() if p.piece_type == chess.BISHOP]
                    if (bishop_squares[0] + bishop_squares[1]) % 2 == 0:  # Same color squares
                        self.show_end_screen("Draw! Kings and same-colored bishops.")
                        return True

        return False

    def draw_timer(self):
        if self.current_player == chess.WHITE:
            remaining_time = self.white_time
            minutes, seconds = map(int, divmod(remaining_time, 60))
            timer_text = f"White's Time: {minutes:02d}:{seconds:02d}"
        else:
            timer_text = "AI is thinking..."
        text_surface = self.font.render(timer_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, 20))
        self.screen.blit(text_surface, text_rect)

    def check_timeout(self):
        if self.timer_start is not None and self.current_player == chess.WHITE:
            current_time = time.time()
            if self.last_time_check is None:
                self.last_time_check = current_time
            elapsed_time = current_time - self.last_time_check
            self.last_time_check = current_time

            new_time = max(0, self.white_time - elapsed_time)
            if new_time == 0:
                self.show_end_screen("White loses! Black wins due to timeout!")
                return True
            self.white_time = new_time
        return False
    def show_start_screen(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if WIDTH // 2 - 100 < x < WIDTH // 2 + 100 and WIDTH // 2 - 50 < y < WIDTH // 2:
                        return True
                    elif WIDTH // 2 - 100 < x < WIDTH // 2 + 100 and WIDTH // 2 + 50 < y < WIDTH // 2 + 100:
                        pygame.quit()
                        return False

            self.screen.fill((0, 0, 0))
            start_text = self.font.render("Start", True, (255, 255, 255))
            exit_text = self.font.render("Exit", True, (255, 255, 255))
            start_rect = start_text.get_rect(center=(WIDTH // 2, WIDTH // 2 - 25))
            exit_rect = exit_text.get_rect(center=(WIDTH // 2, WIDTH // 2 + 75))
            pygame.draw.rect(self.screen, (128, 128, 128), (WIDTH // 2 - 100, WIDTH // 2 - 50, 200, 50))
            pygame.draw.rect(self.screen, (128, 128, 128), (WIDTH // 2 - 100, WIDTH // 2 + 50, 200, 50))
            self.screen.blit(start_text, start_rect)
            self.screen.blit(exit_text, exit_rect)
            pygame.display.flip()

    def show_promotion_choice(self, square):
        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        piece_names = ["Queen", "Rook", "Bishop", "Knight"]
        for i, piece_type in enumerate(promotion_pieces):
            pygame.draw.rect(self.screen, (128, 128, 128),
                           (WIDTH//2-100, WIDTH//2-150+i*50, 200, 50))
            text = self.font.render(piece_names[i], True, (255, 255, 255))
            text_rect = text.get_rect(center=(WIDTH//2, WIDTH//2-125+i*50))
            self.screen.blit(text, text_rect)

    def show_end_screen(self, message):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if WIDTH // 2 - 100 < x < WIDTH // 2 + 100 and WIDTH // 2 - 50 < y < WIDTH // 2:
                        self.reset_game()
                        return
                    elif WIDTH // 2 - 100 < x < WIDTH // 2 + 100 and WIDTH // 2 + 50 < y < WIDTH // 2 + 100:
                        pygame.quit()
                        return

            self.screen.fill((0, 0, 0))
            result_text = self.font.render(message, True, (255, 255, 255))
            keep_text = self.font.render("Keep", True, (255, 255, 255))
            exit_text = self.font.render("Exit", True, (255, 255, 255))
            result_rect = result_text.get_rect(center=(WIDTH // 2, WIDTH // 2 - 100))
            keep_rect = keep_text.get_rect(center=(WIDTH // 2, WIDTH // 2 - 25))
            exit_rect = exit_text.get_rect(center=(WIDTH // 2, WIDTH // 2 + 75))
            pygame.draw.rect(self.screen, (128, 128, 128), (WIDTH // 2 - 100, WIDTH // 2 - 50, 200, 50))
            pygame.draw.rect(self.screen, (128, 128, 128), (WIDTH // 2 - 100, WIDTH // 2 + 50, 200, 50))
            self.screen.blit(result_text, result_rect)
            self.screen.blit(keep_text, keep_rect)
            self.screen.blit(exit_text, exit_rect)
            pygame.display.flip()
    def reset_game(self):
        self.board = chess.Board()
        self.started = False
        self.white_time = TIMEOUT
        self.black_time = TIMEOUT
        self.timer_start = None
        self.last_time_check = None
        self.current_player = chess.WHITE
        self.is_ai_turn = False if self.current_player == chess.WHITE else True
        self.delay_start_time = None

    def evaluate_board(self):
        score = 0
        # Basic scoring: piece values and position values
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                table = PIECE_TABLES[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value + table[square]
                else:
                    # Flip the position table for black
                    score -= value + table[63 - square]

        # Check if in check (highest priority)
        if self.board.is_check():
            if self.board.turn == chess.WHITE:
                score -= 1000  # White is in check, reduce score
            else:
                score += 1000  # Black is in check, increase score

        # Check threatened pieces and protection strategies (second priority)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                attackers = list(self.board.attackers(not piece.color, square))
                defenders = list(self.board.attackers(piece.color, square))
                if attackers:
                    # Calculate the minimum value of attackers and defenders
                    min_attacker_value = min(PIECE_VALUES[self.board.piece_at(att).piece_type] for att in attackers)
                    min_defender_value = float('inf') if not defenders else \
                                       min(PIECE_VALUES[self.board.piece_at(def_).piece_type] for def_ in defenders)

                    # If defender value is less than attacker, it's a good defense
                    if min_defender_value < min_attacker_value:
                        defense_bonus = (min_attacker_value - min_defender_value) * 0.2
                        if piece.color == chess.WHITE:
                            score += defense_bonus
                        else:
                            score -= defense_bonus
                    else:
                        # Penalty for being threatened and unable to defend effectively
                        threat_value = PIECE_VALUES[piece.piece_type] * 0.8
                        if piece.color == chess.WHITE:
                            score -= threat_value
                        else:
                            score += threat_value

        # Check for capturable opponent pieces (third priority)
        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                captured_piece = self.board.piece_at(move.to_square)
                if captured_piece:
                    # Simulate the capture move
                    self.board.push(move)

                    # Check if there will be a recapture after capture
                    attackers = list(self.board.attackers(not self.board.turn, move.to_square))
                    defenders = list(self.board.attackers(self.board.turn, move.to_square))

                    # Calculate the score of the position after capture
                    capture_score = PIECE_VALUES[captured_piece.piece_type]
                    if attackers:
                        # If there will be recapture, consider the loss
                        min_attacker_value = min(PIECE_VALUES[self.board.piece_at(att).piece_type] for att in attackers)
                        if not defenders:
                            # No defenders, will lose the capturing piece
                            capture_score = PIECE_VALUES[captured_piece.piece_type] - \
                                          PIECE_VALUES[self.board.piece_at(move.to_square).piece_type]
                        else:
                            # Has defenders, but evaluate if the exchange is worth it
                            min_defender_value = min(PIECE_VALUES[self.board.piece_at(def_).piece_type] \
                                                    for def_ in defenders)
                            if min_defender_value > min_attacker_value:
                                # Defenders are more valuable than attackers, not worth it
                                capture_score = PIECE_VALUES[captured_piece.piece_type] - \
                                               PIECE_VALUES[self.board.piece_at(move.to_square).piece_type]

                    self.board.pop()

                    # Adjust score based on the position score
                    if self.board.turn == chess.WHITE:
                        score += capture_score * 0.3
                    else:
                        score -= capture_score * 0.3

        return score

    def minimax(self, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.board.is_game_over():
            return self.evaluate_board()

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.board.legal_moves:
                self.board.push(move)
                eval = self.minimax(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.board.legal_moves:
                self.board.push(move)
                eval = self.minimax(depth - 1, alpha, beta, True)
                self.board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def find_best_move(self):
        best_move = None
        max_depth = 4  # Search depth, can be adjusted based on performance
        for depth in range(1, max_depth + 1):
            current_best_move = self._find_best_move_at_depth(depth)
            if current_best_move:
                best_move = current_best_move
        return best_move

    def _find_best_move_at_depth(self, depth):
        best_move = None
        best_eval = float('-inf') if self.board.turn == chess.WHITE else float('inf')

        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.turn == chess.WHITE:
                eval = self.minimax(depth - 1, float('-inf'), float('inf'), False)
            else:
                eval = self.minimax(depth - 1, float('-inf'), float('inf'), True)
            self.board.pop()

            if self.board.turn == chess.WHITE and eval > best_eval:
                best_eval = eval
                best_move = move
            elif self.board.turn == chess.BLACK and eval < best_eval:
                best_eval = eval
                best_move = move

        return best_move

    def run(self):
        if not self.show_start_screen():
            return
        selected_square = None
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.is_ai_turn:
                        x, y = pygame.mouse.get_pos()
                        col = x // SQ_SIZE
                        row = 7 - (y // SQ_SIZE)
                        square = chess.square(col, row)

                        if not self.started:
                            piece = self.board.piece_at(square)
                            if piece and piece.color == chess.WHITE:
                                self.started = True
                                self.timer_start = time.time()
                                self.last_time_check = time.time()

                        if self.started:
                            if selected_square is None:
                                piece = self.board.piece_at(square)
                                if piece and piece.color == self.board.turn:
                                    selected_square = square
                            else:
                                move = chess.Move(selected_square, square)
                                # Check if it is pawn promotion move
                                is_promotion = False
                                piece = self.board.piece_at(selected_square)
                                if piece and piece.piece_type == chess.PAWN:
                                    if (piece.color == chess.WHITE and chess.square_rank(square) == 7) or \
                                       (piece.color == chess.BLACK and chess.square_rank(square) == 0):
                                        is_promotion = True
                                        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                                        self.show_promotion_choice(square)
                                        while self.promotion_choice is None:
                                            pygame.display.flip()
                                            for ev in pygame.event.get():
                                                if ev.type == pygame.MOUSEBUTTONDOWN:
                                                    mx, my = pygame.mouse.get_pos()
                                                    for i, piece_type in enumerate(promotion_pieces):
                                                        if WIDTH//2-100 <= mx <= WIDTH//2+100 and \
                                                           WIDTH//2-150+i*50 <= my <= WIDTH//2-100+i*50:
                                                            self.promotion_choice = piece_type
                                        move = chess.Move(selected_square, square, promotion=self.promotion_choice)
                                        self.promotion_choice = None

                                # Check if it is en passant capture
                                elif piece and piece.piece_type == chess.PAWN and self.last_move:
                                    last_piece = self.board.piece_at(self.last_move.to_square)
                                    if last_piece and last_piece.piece_type == chess.PAWN:
                                        if abs(self.last_move.from_square - self.last_move.to_square) == 16:
                                            if chess.square_file(square) == chess.square_file(self.last_move.to_square):
                                                if abs(chess.square_rank(selected_square) - chess.square_rank(self.last_move.to_square)) == 0:
                                                    ep_square = self.last_move.to_square + (8 if piece.color == chess.WHITE else -8)
                                                    move = chess.Move(selected_square, ep_square)

                                if move in self.board.legal_moves:
                                    self.board.push(move)
                                    self.last_move = move
                                    self.last_move_from = selected_square  # Record starting position
                                    self.current_player = not self.current_player
                                    self.is_ai_turn = True
                                    if self.current_player == chess.WHITE:
                                        self.white_time = TIMEOUT
                                    else:
                                        self.black_time = TIMEOUT
                                    self.timer_start = time.time()
                                    self.last_time_check = time.time()
                                    self.delay_start_time = time.time()
                                if self.promotion_choice:
                                    # Create a new move with promotion info
                                    promotion_move = chess.Move(selected_square, square, promotion=self.promotion_choice)
                                    if promotion_move in self.board.legal_moves:
                                        self.board.push(promotion_move)
                                        self.current_player = not self.current_player
                                        self.is_ai_turn = True
                                        if self.current_player == chess.WHITE:
                                            self.white_time = TIMEOUT
                                        else:
                                            self.black_time = TIMEOUT
                                        self.timer_start = time.time()
                                        self.last_time_check = time.time()
                                        self.last_move = promotion_move
                                        self.last_move_from = selected_square
                                    self.promotion_choice = None
                                selected_square = None
                                new_piece = self.board.piece_at(square)
                                if new_piece and new_piece.color == self.board.turn:
                                    selected_square = square

            if self.is_ai_turn:
                if self.delay_start_time is not None:
                    elapsed_delay = time.time() - self.delay_start_time
                    if elapsed_delay >= 1:
                        best_move = self.find_best_move()
                        if best_move:
                            self.board.push(best_move)
                            self.last_move = best_move  # Update last move
                            self.last_move_from = best_move.from_square  # Record AI move start position
                            self.current_player = not self.current_player
                            self.is_ai_turn = False
                            if self.current_player == chess.WHITE:
                                self.white_time = TIMEOUT
                            self.timer_start = time.time()
                            self.last_time_check = time.time()
                        self.delay_start_time = None

            self.draw_board()
            self.draw_pieces()

            if not self.started:
                start_text = "White starts first. Click a white piece to begin."
                text_surface = self.font.render(start_text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(WIDTH // 2, WIDTH // 2))
                self.screen.blit(text_surface, text_rect)
            else:
                if self.check_timeout():
                    continue
                if self.check_checkmate():
                    continue

                if selected_square is not None and not self.is_ai_turn:
                    valid_moves = [move for move in self.board.legal_moves if move.from_square == selected_square]
                    self.draw_valid_moves(valid_moves)

                self.draw_timer()

            pygame.display.flip()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()