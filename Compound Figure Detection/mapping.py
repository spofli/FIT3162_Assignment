def ri_mapping(P):
    table = [0] * (2 ** P)
    temp_table = [-1] * (2 ** P)
    map_val = 0
    for i in range(2 ** P):
        lowest = i
        left_rotate = i
        for j in range(P - 1):  # rotate P times and get lowest value
            left_rotate = (left_rotate << 1) & (2 ** P - 1) | (left_rotate & (2 ** P - 1)) >> (P - 1)
            if left_rotate < lowest:
                lowest = left_rotate
        if temp_table[lowest] < 0:  # set lowest value or use previously obtained value
            temp_table[lowest] = map_val
            map_val += 1
        table[i] = temp_table[lowest]
    bins = map_val
    return table, bins


def u2_mapping(P):
    table = [0] * 2**P
    index = 0
    bins = (P * (P - 1)) + 3
    for i in range(2**P):
        transitions = 0
        left_rotate = (i << 1) & (2**P - 1) | (i & (2**P - 1)) >> (P - 1)  # rotate bits left
        xor = i ^ left_rotate  # xor original bits with rotated bits
        # sum of bits = number of transitions
        while xor != 0: # sum bits
            xor &= xor - 1
            transitions += 1
        # set mapping table
        if transitions <= 2:
            table[i] = index
            index += 1
        else:
            table[i] = (P * (P - 1)) + 2
    return table, bins


def riu2_mapping(P):
    table = [0] * 2**P
    bins = P + 2
    for i in range(2**P):
        transitions = 0
        left_rotate = (i << 1) & (2**P - 1) | (i & (2**P - 1)) >> (P - 1)  # rotate bits left
        xor = i ^ left_rotate  # xor original bits with rotated bits
        # sum of bits = number of transitions
        while xor != 0: # sum bits
            xor &= xor - 1
            transitions += 1
        # set mapping table
        if transitions <= 2:
            val = i
            while val != 0:  # sum bits
                val &= val - 1
                table[i] += 1
        else:
            table[i] = P + 1
    return table, bins


def nr_mapping(P):
    table = [0] * 2**P
    bins = 2**P // 2
    for i in range(2**P):
        table[i] = min(i, (2 ** P) - 1 - i)  # half of the values are the same
    return table, bins


def nrriu2_mapping(P):
    table = [0] * 2 ** P
    bins = P + 1
    for i in range(2 ** P):
        new_i = min(i, (2 ** P) - 1 - i)
        transitions = 0
        left_rotate = (new_i << 1) & (2**P - 1) | (new_i & (2**P - 1)) >> (P - 1)  # rotate bits left
        xor = new_i ^ left_rotate  # xor original bits with rotated bits
        # sum of bits = number of transitions
        while xor != 0: # sum bits
            xor &= xor - 1
            transitions += 1
        # set mapping table
        if transitions <= 2:
            val = new_i
            while val != 0:  # sum bits
                val &= val - 1
                table[i] += 1
        else:
            table[i] = P
    return table, bins


