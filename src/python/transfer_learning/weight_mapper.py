"""
Weight Mapper для автоматического mapping весов GGUF → ExpertModel.

Основные возможности:
    - Автоматический mapping имён тензоров
    - Layer-by-layer transfer для экономии памяти
    - Resize embeddings при несовместимости vocab_size
    - Обработка различных архитектур (Llama, Phi, Mistral)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path

from .gguf_parser import GGUFParser
from .memory_manager import MemoryManager


class WeightMapper:
    """
    Автоматический mapping весов из GGUF модели в ExpertModel.

    Создаёт соответствие между именами тензоров в GGUF и PyTorch модели,
    затем переносит веса layer-by-layer для эффективного использования памяти.

    Пример:
        >>> parser = GGUFParser("phi-3-mini-q8.gguf")
        >>> model = ExpertModel(vocab_size=8000, d_model=512, n_layers=6)
        >>>
        >>> mapper = WeightMapper(parser, model)
        >>> mapping = mapper.create_mapping()
        >>> print(f"Найдено {len(mapping)} соответствий")
        >>>
        >>> report = mapper.transfer_weights(layers_to_transfer=[0, 1, 2, 3])
        >>> print(f"Перенесено: {len(report['transferred'])}")
    """

    def __init__(
        self,
        gguf_parser: GGUFParser,
        expert_model: nn.Module,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        Инициализирует Weight Mapper.

        Args:
            gguf_parser: GGUF parser для чтения тензоров
            expert_model: Целевая ExpertModel
            memory_manager: Опциональный memory manager для контроля RAM
        """
        self.gguf_parser = gguf_parser
        self.expert_model = expert_model
        self.memory_manager = memory_manager or MemoryManager()

        # Извлекаем metadata
        self.gguf_metadata = gguf_parser.get_metadata()
        self.gguf_tensors = gguf_parser.list_tensors()

        print(f"✅ WeightMapper инициализирован:")
        print(f"   GGUF тензоров: {len(self.gguf_tensors)}")
        print(f"   Архитектура GGUF: {self.gguf_metadata.get('architecture', 'unknown')}")

    def create_mapping(self) -> Dict[str, str]:
        """
        Создаёт автоматический mapping имён тензоров GGUF → ExpertModel.

        Returns:
            Словарь: {gguf_tensor_name: expert_model_param_name}

        Example:
            >>> mapping = mapper.create_mapping()
            >>> for gguf_name, model_name in mapping.items():
            >>>     print(f"{gguf_name} → {model_name}")
        """
        mapping = {}

        # Получаем state dict ExpertModel для определения целевых имён
        model_state_dict = self.expert_model.state_dict()
        model_params = set(model_state_dict.keys())

        # Определяем архитектуру из GGUF
        arch = self.gguf_metadata.get('architecture', 'llama')

        # 1. Mapping для token embeddings
        emb_mapping = self._map_embeddings(arch)
        mapping.update(emb_mapping)

        # 2. Mapping для transformer блоков
        layers_mapping = self._map_transformer_layers(arch)
        mapping.update(layers_mapping)

        # 3. Mapping для output projection (lm_head)
        output_mapping = self._map_output_head(arch)
        mapping.update(output_mapping)

        # Фильтруем только существующие в модели параметры
        valid_mapping = {}
        for gguf_name, model_name in mapping.items():
            if model_name in model_params:
                valid_mapping[gguf_name] = model_name
            else:
                print(f"⚠️  Параметр '{model_name}' не найден в модели, пропускаем")

        print(f"✅ Создан mapping: {len(valid_mapping)} соответствий")
        return valid_mapping

    def _map_embeddings(self, arch: str) -> Dict[str, str]:
        """
        Mapping для token embeddings.

        Args:
            arch: Архитектура GGUF (llama, phi, mistral, etc.)

        Returns:
            Словарь mapping для embeddings
        """
        mapping = {}

        # Llama/Mistral style: "token_embd.weight"
        if "token_embd.weight" in self.gguf_tensors:
            mapping["token_embd.weight"] = "token_embedding.weight"

        # Phi style: "model.embed_tokens.weight"
        elif "model.embed_tokens.weight" in self.gguf_tensors:
            mapping["model.embed_tokens.weight"] = "token_embedding.weight"

        # GPT style: "wte.weight"
        elif "wte.weight" in self.gguf_tensors:
            mapping["wte.weight"] = "token_embedding.weight"

        return mapping

    def _map_transformer_layers(self, arch: str) -> Dict[str, str]:
        """
        Mapping для transformer слоёв.

        Args:
            arch: Архитектура GGUF

        Returns:
            Словарь mapping для всех transformer слоёв
        """
        mapping = {}

        # Определяем количество слоёв
        n_layers_gguf = self.gguf_metadata.get('n_layers', 32)
        n_layers_model = getattr(self.expert_model, 'n_layers', 6)

        # Переносим только доступное количество слоёв
        n_layers_to_map = min(n_layers_gguf, n_layers_model)

        for layer_idx in range(n_layers_to_map):
            # Llama/Mistral style: "blk.{i}.*"
            layer_mapping = self._map_single_layer_llama_style(layer_idx)
            if layer_mapping:
                mapping.update(layer_mapping)
                continue

            # Phi style: "model.layers.{i}.*"
            layer_mapping = self._map_single_layer_phi_style(layer_idx)
            if layer_mapping:
                mapping.update(layer_mapping)
                continue

        return mapping

    def _map_single_layer_llama_style(self, layer_idx: int) -> Dict[str, str]:
        """
        Mapping для одного transformer слоя (Llama/Mistral style).

        Args:
            layer_idx: Индекс слоя

        Returns:
            Словарь mapping для этого слоя
        """
        mapping = {}
        prefix_gguf = f"blk.{layer_idx}"
        prefix_model = f"transformer_blocks.{layer_idx}"

        # Attention weights
        # Q, K, V, O projections
        attn_mappings = {
            f"{prefix_gguf}.attn_q.weight": f"{prefix_model}.attention.W_q.weight",
            f"{prefix_gguf}.attn_k.weight": f"{prefix_model}.attention.W_k.weight",
            f"{prefix_gguf}.attn_v.weight": f"{prefix_model}.attention.W_v.weight",
            f"{prefix_gguf}.attn_output.weight": f"{prefix_model}.attention.W_o.weight",

            # Attention biases (если есть)
            f"{prefix_gguf}.attn_q.bias": f"{prefix_model}.attention.W_q.bias",
            f"{prefix_gguf}.attn_k.bias": f"{prefix_model}.attention.W_k.bias",
            f"{prefix_gguf}.attn_v.bias": f"{prefix_model}.attention.W_v.bias",
            f"{prefix_gguf}.attn_output.bias": f"{prefix_model}.attention.W_o.bias",
        }

        # FFN weights
        ffn_mappings = {
            f"{prefix_gguf}.ffn_up.weight": f"{prefix_model}.ffn.linear1.weight",
            f"{prefix_gguf}.ffn_down.weight": f"{prefix_model}.ffn.linear2.weight",

            # FFN biases (если есть)
            f"{prefix_gguf}.ffn_up.bias": f"{prefix_model}.ffn.linear1.bias",
            f"{prefix_gguf}.ffn_down.bias": f"{prefix_model}.ffn.linear2.bias",
        }

        # LayerNorm weights
        norm_mappings = {
            f"{prefix_gguf}.attn_norm.weight": f"{prefix_model}.norm1.weight",
            f"{prefix_gguf}.attn_norm.bias": f"{prefix_model}.norm1.bias",
            f"{prefix_gguf}.ffn_norm.weight": f"{prefix_model}.norm2.weight",
            f"{prefix_gguf}.ffn_norm.bias": f"{prefix_model}.norm2.bias",
        }

        # Объединяем все mappings
        all_mappings = {**attn_mappings, **ffn_mappings, **norm_mappings}

        # Добавляем только существующие тензоры
        for gguf_name, model_name in all_mappings.items():
            if gguf_name in self.gguf_tensors:
                mapping[gguf_name] = model_name

        return mapping

    def _map_single_layer_phi_style(self, layer_idx: int) -> Dict[str, str]:
        """
        Mapping для одного transformer слоя (Phi style).

        Args:
            layer_idx: Индекс слоя

        Returns:
            Словарь mapping для этого слоя
        """
        mapping = {}
        prefix_gguf = f"model.layers.{layer_idx}"
        prefix_model = f"transformer_blocks.{layer_idx}"

        # Phi может использовать другие имена
        # TODO: Добавить support для Phi naming convention

        return mapping

    def _map_output_head(self, arch: str) -> Dict[str, str]:
        """
        Mapping для output projection (lm_head).

        Args:
            arch: Архитектура GGUF

        Returns:
            Словарь mapping для lm_head
        """
        mapping = {}

        # Llama/Mistral style: "output.weight"
        if "output.weight" in self.gguf_tensors:
            mapping["output.weight"] = "lm_head.weight"

        # Phi style: "lm_head.weight" (уже совпадает)
        elif "lm_head.weight" in self.gguf_tensors:
            mapping["lm_head.weight"] = "lm_head.weight"

        return mapping

    def transfer_weights(
        self,
        layers_to_transfer: Optional[List[int]] = None,
        freeze_layers: bool = False,
        resize_embeddings: bool = True
    ) -> Dict[str, List[str]]:
        """
        Переносит веса из GGUF в ExpertModel layer-by-layer.

        Args:
            layers_to_transfer: Список индексов слоёв для переноса (None = все)
            freeze_layers: Заморозить перенесённые веса (не обучать)
            resize_embeddings: Автоматически resize embeddings если vocab mismatch

        Returns:
            Отчёт о переносе:
                - 'transferred': список успешно перенесённых параметров
                - 'skipped': список пропущенных параметров
                - 'resized': список параметров с resize
                - 'frozen': список замороженных параметров

        Example:
            >>> report = mapper.transfer_weights(
            ...     layers_to_transfer=[0, 1, 2, 3],
            ...     freeze_layers=True
            ... )
            >>> print(f"Перенесено: {len(report['transferred'])}")
            >>> print(f"Пропущено: {len(report['skipped'])}")
        """
        # Создаём mapping
        mapping = self.create_mapping()

        report = {
            'transferred': [],
            'skipped': [],
            'resized': [],
            'frozen': []
        }

        # Получаем state dict модели
        model_state_dict = self.expert_model.state_dict()

        print(f"\n🔄 Начинаем перенос весов...")
        print(f"   Всего соответствий: {len(mapping)}")

        # Переносим веса
        for gguf_name, model_name in mapping.items():
            try:
                # Фильтрация по слоям если указано
                if layers_to_transfer is not None:
                    # Проверяем принадлежность к нужному слою
                    if not self._should_transfer_param(model_name, layers_to_transfer):
                        report['skipped'].append(model_name)
                        continue

                # Проверяем память перед загрузкой
                tensor_info = self.gguf_parser.get_tensor_info(gguf_name)
                tensor_size_bytes = tensor_info['n_elements'] * 4  # FP32

                if not self.memory_manager.can_load_tensor_size(tensor_size_bytes):
                    print(f"⚠️  Недостаточно памяти для {gguf_name}, пропускаем")
                    report['skipped'].append(model_name)
                    continue

                # Загружаем тензор из GGUF
                gguf_tensor = self.gguf_parser.load_tensor(gguf_name, dequantize=True)

                # Получаем целевую форму из модели
                target_param = model_state_dict[model_name]
                target_shape = target_param.shape

                # Проверяем совместимость размерностей
                if gguf_tensor.shape != target_shape:
                    # Пытаемся resize если это embeddings
                    if 'embedding' in model_name or 'lm_head' in model_name:
                        if resize_embeddings:
                            gguf_tensor = self._resize_embedding_tensor(
                                gguf_tensor,
                                target_shape,
                                model_name
                            )
                            report['resized'].append(model_name)
                        else:
                            print(f"⚠️  Размерности не совпадают для {model_name}")
                            print(f"     GGUF: {gguf_tensor.shape}, Model: {target_shape}")
                            report['skipped'].append(model_name)
                            continue
                    else:
                        print(f"⚠️  Размерности не совпадают для {model_name}")
                        print(f"     GGUF: {gguf_tensor.shape}, Model: {target_shape}")
                        report['skipped'].append(model_name)
                        continue

                # Копируем веса в модель
                with torch.no_grad():
                    model_state_dict[model_name].copy_(gguf_tensor)

                report['transferred'].append(model_name)
                print(f"✅ Перенесён: {model_name}")

            except Exception as e:
                print(f"❌ Ошибка при переносе {gguf_name}: {str(e)}")
                report['skipped'].append(model_name)

        # Загружаем обновлённый state dict
        self.expert_model.load_state_dict(model_state_dict)

        # Замораживаем веса если нужно
        if freeze_layers:
            frozen = self._freeze_transferred_params(report['transferred'])
            report['frozen'] = frozen

        print(f"\n✅ Перенос завершён!")
        print(f"   Перенесено: {len(report['transferred'])}")
        print(f"   Пропущено: {len(report['skipped'])}")
        print(f"   Resized: {len(report['resized'])}")
        print(f"   Заморожено: {len(report['frozen'])}")

        return report

    def _should_transfer_param(self, param_name: str, layers_to_transfer: List[int]) -> bool:
        """
        Проверяет, нужно ли переносить параметр на основе списка слоёв.

        Args:
            param_name: Имя параметра в модели
            layers_to_transfer: Список индексов слоёв для переноса

        Returns:
            True если параметр нужно переносить
        """
        # Embeddings и lm_head всегда переносим
        if 'embedding' in param_name or 'lm_head' in param_name:
            return True

        # Для transformer блоков проверяем индекс
        if 'transformer_blocks' in param_name:
            # Извлекаем индекс слоя: transformer_blocks.{idx}.*
            parts = param_name.split('.')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    return layer_idx in layers_to_transfer
                except ValueError:
                    pass

        return True

    def _resize_embedding_tensor(
        self,
        source_tensor: torch.Tensor,
        target_shape: tuple,
        param_name: str
    ) -> torch.Tensor:
        """
        Resize embedding тензора при несовпадении vocab_size.

        Args:
            source_tensor: Исходный тензор из GGUF
            target_shape: Целевая форма
            param_name: Имя параметра (для логирования)

        Returns:
            Resized тензор
        """
        print(f"🔧 Resize {param_name}: {source_tensor.shape} → {target_shape}")

        # Для embeddings: [vocab_size, d_model]
        if len(source_tensor.shape) == 2 and len(target_shape) == 2:
            source_vocab, source_dim = source_tensor.shape
            target_vocab, target_dim = target_shape

            # Если d_model совпадает, просто обрезаем/дополняем vocabulary
            if source_dim == target_dim:
                if target_vocab < source_vocab:
                    # Обрезаем
                    return source_tensor[:target_vocab, :]
                else:
                    # Дополняем случайной инициализацией
                    new_tensor = torch.randn(target_vocab, target_dim) * 0.02
                    new_tensor[:source_vocab, :] = source_tensor
                    return new_tensor

        # Fallback: просто обрезаем или дополняем
        new_tensor = torch.randn(*target_shape) * 0.02

        # Копируем общую часть
        slices = tuple(slice(0, min(s, t)) for s, t in zip(source_tensor.shape, target_shape))
        new_tensor[slices] = source_tensor[slices]

        return new_tensor

    def _freeze_transferred_params(self, param_names: List[str]) -> List[str]:
        """
        Замораживает перенесённые параметры (requires_grad = False).

        Args:
            param_names: Список имён параметров для заморозки

        Returns:
            Список замороженных параметров
        """
        frozen = []

        for name, param in self.expert_model.named_parameters():
            if name in param_names:
                param.requires_grad = False
                frozen.append(name)

        return frozen
