def compute(operation: str, values: list[dict]) -> float | dict:
    """
    Pure Python arithmetic — the deterministic half of Calculator.
    The LLM extraction step only identifies the operation and the labeled
    numbers involved; the actual math always happens here, never in the
    LLM call, since LLMs are unreliable at precise arithmetic.

    `values` is a list of {"label": str, "value": float} dicts.

    Order matters for percent_change, difference, ratio, and margin:
    values[0] is base/older/denominator, values[1] is new/comparison/
    numerator (or subtract-amount, for margin).

    Returns a float for percent_change/difference/sum/average/ratio/margin.
    Returns a {"label": str, "value": float} dict for max/min, since the
    caller needs to know WHICH item won, not just the winning number.
    """
    nums = [v["value"] for v in values]

    if operation == "percent_change":
        old, new = nums[0], nums[1]
        if old == 0:
            raise ValueError("Cannot compute percent change from a base of 0")
        return round((new - old) / old * 100, 2)

    elif operation == "difference":
        return round(nums[0] - nums[1], 2)

    elif operation == "sum":
        return round(sum(nums), 2)

    elif operation == "average":
        return round(sum(nums) / len(nums), 2)

    elif operation == "ratio":
        if nums[1] == 0:
            raise ValueError("Cannot compute ratio with a denominator of 0")
        return round(nums[0] / nums[1], 4)

    elif operation == "margin":
        # e.g. gross margin = (revenue - cogs) / revenue * 100
        base, subtract = nums[0], nums[1]
        if base == 0:
            raise ValueError("Cannot compute margin with a base of 0")
        return round((base - subtract) / base * 100, 2)

    elif operation in ("max", "min"):
        picker = max if operation == "max" else min
        return picker(values, key=lambda v: v["value"])

    else:
        raise ValueError(f"Unsupported operation: {operation}")