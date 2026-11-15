from typing import Any, Iterator

from .args import DataArgs
from .dataset import build_dataset
from .interleaver import Batch


def build_data_loader(
    instruct_tokenizer: Any,
    args: DataArgs,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
) -> Iterator[Batch]:
    if is_eval:
        assert args.eval_data != "", "No eval data provided."
    pretrain_data = args.train_data if not is_eval else args.eval_data

    dataset = build_dataset(
        pretrain_data=pretrain_data,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle_pretrain=args.shuffle,
    )

    sample_list = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)

        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []

    if is_eval and sample_list:
        yield Batch.collate(sample_list)
