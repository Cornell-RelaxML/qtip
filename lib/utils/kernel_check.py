def has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y):
    if decode_mode != 'quantlut_sym':
        return False
    if L != 16:
        return False
    if V != 2:
        return False
    if K < 2 or K > 4:
        return False
    if tlut_bits != 9:
        return False
    if td_x != 16 or td_y != 16:
        return False
    return True
