"""
Тесты для ExpertModel - полная language model.

Тестирует:
    - PositionalEncoding
    - ExpertModel initialization
    - Forward pass
    - Text generation
    - Save/Load
    - Различные конфигурации
"""

import pytest
import torch
import tempfile
from pathlib import Path
from models.expert import PositionalEncoding, ExpertModel


# ============================================================================
# Тесты для PositionalEncoding
# ============================================================================

@pytest.mark.unit
@pytest.mark.expert
@pytest.mark.fast
class TestPositionalEncoding:
    """Тесты для синусоидального positional encoding."""

    def test_initialization(self, d_model, max_seq_len):
        """Тест инициализации PositionalEncoding."""
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)

        assert hasattr(pe, 'pe')
        # PE buffer должен иметь размерность [1, max_seq_len, d_model]
        assert pe.pe.shape == (1, max_seq_len, d_model)

    def test_forward_shape(self, d_model, max_seq_len, batch_size):
        """Тест размерности выхода."""
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)

        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_adds_positional_info(self, d_model, max_seq_len, batch_size):
        """Тест что PE действительно добавляется к входу."""
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, dropout=0.0)

        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        # Выход должен отличаться от входа (PE добавлен)
        assert not torch.allclose(output, x)

    def test_deterministic(self, d_model, max_seq_len, batch_size):
        """Тест детерминированности PE."""
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, dropout=0.0)
        pe.eval()

        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)

        output1 = pe(x)
        output2 = pe(x)

        # PE детерминирован (без dropout)
        assert torch.allclose(output1, output2)

    def test_position_uniqueness(self, d_model, max_seq_len):
        """Тест что разные позиции имеют разные encodings."""
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)

        # Берём encodings для первых 10 позиций
        encodings = pe.pe[0, :10, :]  # [10, d_model]

        # Проверяем что позиции отличаются друг от друга
        for i in range(10):
            for j in range(i + 1, 10):
                # Разные позиции должны иметь разные encodings
                assert not torch.allclose(encodings[i], encodings[j])


# ============================================================================
# Тесты для ExpertModel
# ============================================================================

@pytest.mark.unit
@pytest.mark.expert
@pytest.mark.fast
class TestExpertModel:
    """Тесты для полной ExpertModel."""

    def test_initialization(self, expert_model):
        """Тест инициализации ExpertModel."""
        assert hasattr(expert_model, 'token_embedding')
        assert hasattr(expert_model, 'pos_encoding')
        assert hasattr(expert_model, 'transformer_blocks')
        assert hasattr(expert_model, 'final_norm')
        assert hasattr(expert_model, 'lm_head')

    def test_forward_shape(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест размерности выхода."""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = expert_model(input_ids)

        # Logits: [batch_size, seq_len, vocab_size]
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_forward_valid_logits(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест что logits валидны."""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = expert_model(input_ids)

        # Logits не должны содержать NaN или Inf
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_different_inputs_different_outputs(self, expert_model, vocab_size, seq_len):
        """Тест что разные входы дают разные выходы."""
        input1 = torch.randint(0, vocab_size, (1, seq_len))
        input2 = torch.randint(0, vocab_size, (1, seq_len))

        # Убеждаемся что входы разные
        assert not torch.equal(input1, input2)

        output1 = expert_model(input1)
        output2 = expert_model(input2)

        # Выходы должны отличаться
        assert not torch.allclose(output1, output2)

    def test_parameter_count(self, count_parameters):
        """Тест количества параметров для разных конфигураций."""
        # Tiny model
        tiny_model = ExpertModel(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_seq_len=64
        )
        tiny_params = count_parameters(tiny_model)
        assert tiny_params > 0
        print(f"\nTiny model: {tiny_params:,} parameters")

        # Small model
        small_model = ExpertModel(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            max_seq_len=128
        )
        small_params = count_parameters(small_model)
        assert small_params > tiny_params
        print(f"Small model: {small_params:,} parameters")

    def test_gradient_flow(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест прохождения градиентов через всю модель."""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward
        logits = expert_model(input_ids)

        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        # Backward
        loss.backward()

        # Проверяем что градиенты есть у всех параметров
        for name, param in expert_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_training_mode(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест training vs eval mode."""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Training mode
        expert_model.train()
        output_train = expert_model(input_ids)

        # Eval mode
        expert_model.eval()
        output_eval = expert_model(input_ids)

        # Должны отличаться (dropout)
        assert not torch.allclose(output_train, output_eval, rtol=1e-5)

    def test_deterministic_eval(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест детерминированности в eval mode."""
        expert_model.eval()

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output1 = expert_model(input_ids)
        output2 = expert_model(input_ids)

        # В eval mode должно быть детерминированным
        assert torch.allclose(output1, output2)


# ============================================================================
# Тесты для генерации текста
# ============================================================================

@pytest.mark.unit
@pytest.mark.expert
@pytest.mark.fast
class TestExpertModelGeneration:
    """Тесты для генерации текста."""

    def test_generate_basic(self, expert_model, vocab_size):
        """Тест базовой генерации."""
        expert_model.eval()

        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = expert_model.generate(prompt, max_new_tokens=10)

        # Должно быть 5 (prompt) + 10 (новых) = 15 токенов
        assert generated.shape == (1, 15)

    def test_generate_temperature(self, expert_model, vocab_size):
        """Тест генерации с разными temperature."""
        expert_model.eval()

        prompt = torch.randint(0, vocab_size, (1, 5))

        # Low temperature (более детерминированно)
        gen_low = expert_model.generate(prompt, max_new_tokens=10, temperature=0.1)

        # High temperature (более случайно)
        gen_high = expert_model.generate(prompt, max_new_tokens=10, temperature=2.0)

        assert gen_low.shape == (1, 15)
        assert gen_high.shape == (1, 15)

        # Оба должны начинаться с prompt
        assert torch.equal(gen_low[:, :5], prompt)
        assert torch.equal(gen_high[:, :5], prompt)

    def test_generate_top_k(self, expert_model, vocab_size):
        """Тест генерации с top-k sampling."""
        expert_model.eval()

        prompt = torch.randint(0, vocab_size, (1, 5))

        generated = expert_model.generate(
            prompt,
            max_new_tokens=10,
            top_k=50,
            temperature=1.0
        )

        assert generated.shape == (1, 15)
        assert torch.equal(generated[:, :5], prompt)

    def test_generate_top_p(self, expert_model, vocab_size):
        """Тест генерации с nucleus (top-p) sampling."""
        expert_model.eval()

        prompt = torch.randint(0, vocab_size, (1, 5))

        generated = expert_model.generate(
            prompt,
            max_new_tokens=10,
            top_p=0.9,
            temperature=1.0
        )

        assert generated.shape == (1, 15)
        assert torch.equal(generated[:, :5], prompt)

    def test_generate_no_repeat(self, expert_model, vocab_size):
        """Тест что генерация не зацикливается на одном токене."""
        expert_model.eval()

        prompt = torch.randint(0, vocab_size, (1, 5))

        generated = expert_model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50
        )

        # Проверяем что не все новые токены одинаковые
        new_tokens = generated[0, 5:]
        unique_tokens = torch.unique(new_tokens)

        # Должно быть хотя бы 2 разных токена (не зациклен)
        assert len(unique_tokens) >= 2


# ============================================================================
# Тесты для Save/Load
# ============================================================================

@pytest.mark.unit
@pytest.mark.expert
@pytest.mark.fast
class TestExpertModelSaveLoad:
    """Тесты для сохранения и загрузки модели."""

    def test_save_checkpoint(self, expert_model):
        """Тест сохранения checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"

            # Используем стандартный torch.save
            torch.save(expert_model.state_dict(), save_path)

            assert save_path.exists()

    def test_load_checkpoint(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
        """Тест загрузки checkpoint."""
        # Создаём модель
        model1 = ExpertModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"

            # Сохраняем
            torch.save(model1.state_dict(), save_path)

            # Создаём новую модель и загружаем
            model2 = ExpertModel(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len
            )
            model2.load_state_dict(torch.load(save_path))

            # Проверяем что параметры идентичны
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)

    def test_save_load_preserves_output(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест что save/load сохраняет выходы модели."""
        expert_model.eval()

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Выход до сохранения
        output_before = expert_model(input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"

            # Сохраняем
            torch.save(expert_model.state_dict(), save_path)

            # Загружаем в ту же модель
            expert_model.load_state_dict(torch.load(save_path))

            # Выход после загрузки
            output_after = expert_model(input_ids)

            # Должны быть идентичны
            assert torch.allclose(output_before, output_after)

    def test_get_config(self, expert_model, vocab_size, d_model, n_layers, max_seq_len):
        """Тест получения конфигурации модели (хранится как атрибуты)."""
        # Конфигурация хранится как атрибуты модели
        assert expert_model.vocab_size == vocab_size
        assert expert_model.d_model == d_model
        assert expert_model.n_layers == n_layers
        assert expert_model.max_seq_len == max_seq_len


# ============================================================================
# Тесты для различных конфигураций моделей
# ============================================================================

@pytest.mark.unit
@pytest.mark.expert
class TestExpertModelConfigurations:
    """Тесты для различных конфигураций ExpertModel."""

    @pytest.mark.parametrize("config_name,params", [
        ("tiny", {"vocab_size": 1000, "d_model": 128, "n_layers": 2, "n_heads": 4, "d_ff": 512, "max_seq_len": 64}),
        ("small", {"vocab_size": 1000, "d_model": 256, "n_layers": 4, "n_heads": 8, "d_ff": 1024, "max_seq_len": 128}),
        ("medium", {"vocab_size": 5000, "d_model": 512, "n_layers": 8, "n_heads": 8, "d_ff": 2048, "max_seq_len": 256}),
    ])
    def test_various_configs(self, config_name, params):
        """Тест различных конфигураций модели."""
        model = ExpertModel(**params)

        # Тест forward pass
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, params['vocab_size'], (batch_size, seq_len))

        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, params['vocab_size'])
        assert not torch.isnan(logits).any()

        print(f"\n{config_name.upper()} config:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Forward pass: ✅")

    def test_extreme_small_model(self):
        """Тест очень маленькой модели."""
        model = ExpertModel(
            vocab_size=100,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            max_seq_len=16
        )

        input_ids = torch.randint(0, 100, (1, 8))
        logits = model(input_ids)

        assert logits.shape == (1, 8, 100)
        assert not torch.isnan(logits).any()

    def test_varying_sequence_lengths(self, expert_model, vocab_size):
        """Тест с разными длинами последовательностей."""
        expert_model.eval()

        for seq_len in [1, 5, 10, 20, 32]:
            input_ids = torch.randint(0, vocab_size, (1, seq_len))
            logits = expert_model(input_ids)

            assert logits.shape == (1, seq_len, vocab_size)
            assert not torch.isnan(logits).any()


# ============================================================================
# Integration тесты
# ============================================================================

@pytest.mark.integration
@pytest.mark.expert
class TestExpertModelIntegration:
    """Integration тесты для ExpertModel."""

    def test_full_training_step(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест полного шага обучения."""
        import torch.optim as optim

        optimizer = optim.Adam(expert_model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward
        logits = expert_model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss должен быть конечным
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() > 0

    def test_multiple_training_steps(self, expert_model, batch_size, seq_len, vocab_size):
        """Тест нескольких шагов обучения."""
        import torch.optim as optim

        optimizer = optim.Adam(expert_model.parameters(), lr=1e-3)

        losses = []

        for _ in range(5):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, vocab_size, (batch_size, seq_len))

            logits = expert_model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Все loss должны быть валидными
        for loss in losses:
            assert not torch.isnan(torch.tensor(loss))
            assert loss > 0

        print(f"\nLosses over 5 steps: {losses}")

    def test_generation_after_training(self, expert_model, vocab_size):
        """Тест генерации после обучения."""
        import torch.optim as optim

        optimizer = optim.Adam(expert_model.parameters(), lr=1e-3)

        # Несколько шагов обучения
        for _ in range(3):
            input_ids = torch.randint(0, vocab_size, (2, 10))
            targets = torch.randint(0, vocab_size, (2, 10))

            logits = expert_model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Генерация после обучения
        expert_model.eval()
        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = expert_model.generate(prompt, max_new_tokens=10)

        assert generated.shape == (1, 15)
        assert torch.equal(generated[:, :5], prompt)
