def split_part_by_cp_tp(cp_size, cp_rank, tp_size, tp_rank):
    split_size = tp_size
    if cp_size > 1:
        split_size *= cp_size * 2

    part_list = list(range(split_size))

    cp_rank2 = 2 * cp_size - cp_rank - 1
    cp_part_list = (
        part_list[cp_rank * tp_size : (cp_rank + 1) * tp_size]
        + part_list[cp_rank2 * tp_size : (cp_rank2 + 1) * tp_size]
    )

    assert len(cp_part_list) % tp_size == 0
    echo_tp_len = len(cp_part_list) // tp_size
    cp_tp_part_list = cp_part_list[tp_rank * echo_tp_len : (tp_rank + 1) * echo_tp_len]
    return cp_tp_part_list


if __name__ == "__main__":
    print(f"{split_part_by_cp_tp(1, 0, 2, 0)=}")
    print(f"{split_part_by_cp_tp(1, 0, 2, 1)=}")
    print(f"========")
    print(f"{split_part_by_cp_tp(2, 0, 1, 0)=}")
    print(f"{split_part_by_cp_tp(2, 1, 1, 0)=}")
    print(f"========")
    print(f"{split_part_by_cp_tp(2, 0, 2, 0)=}")
    print(f"{split_part_by_cp_tp(2, 0, 2, 1)=}")
    print(f"{split_part_by_cp_tp(2, 1, 2, 0)=}")
    print(f"{split_part_by_cp_tp(2, 1, 2, 1)=}")
    print(f"========")
    print(f"{split_part_by_cp_tp(4, 0, 2, 0)=}")
    print(f"{split_part_by_cp_tp(4, 0, 2, 1)=}")
    print(f"{split_part_by_cp_tp(4, 1, 2, 0)=}")
    print(f"{split_part_by_cp_tp(4, 1, 2, 1)=}")
    print(f"{split_part_by_cp_tp(4, 2, 2, 0)=}")
    print(f"{split_part_by_cp_tp(4, 2, 2, 1)=}")
    print(f"{split_part_by_cp_tp(4, 3, 2, 0)=}")
    print(f"{split_part_by_cp_tp(4, 3, 2, 1)=}")
    print(f"========")
    print(f"{split_part_by_cp_tp(2, 0, 4, 0)=}")
    print(f"{split_part_by_cp_tp(2, 0, 4, 1)=}")
    print(f"{split_part_by_cp_tp(2, 0, 4, 2)=}")
    print(f"{split_part_by_cp_tp(2, 0, 4, 3)=}")
    print(f"{split_part_by_cp_tp(2, 1, 4, 0)=}")
    print(f"{split_part_by_cp_tp(2, 1, 4, 1)=}")
    print(f"{split_part_by_cp_tp(2, 1, 4, 2)=}")
    print(f"{split_part_by_cp_tp(2, 1, 4, 3)=}")
    print(f"========")
