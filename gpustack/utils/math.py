def largest_power_of_2_leq(n: int) -> int:
    """Return the largest power of 2 less than or equal to n."""

    if n < 1:
        return 0

    return 1 << (n.bit_length() - 1)
