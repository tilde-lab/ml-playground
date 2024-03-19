from ase.data import chemical_symbols

periodic_numbers = [0,
                                                                                                                                                                                    1,    112,
2,    8,                                                                                                                                                     82,   88,   94,  100,  106,  113,
3,    9,                                                                                                                                                     83,   89,   95,  101,  107,  114,
4,   10,                                                                                        14,    46,   50,   54,   58,   62,   66,   70,   74,   78,   84,   90,   96,  102,  108,  115,
5,   11,                                                                                        15,    47,   51,   55,   59,   63,   67,   71,   75,   79,   85,   91,   97,  103,  109,  116,
6,   12,    16,    18,   20,  22,   24,    26,   28,   30,   32,   34,   36,   38,   40,  42,   44,    48,   52,   56,   60,   64,   68,   72,   76,   80,   86,   92,   98,  104,  110,  117,
7,   13,    17,    19,   21,  23,   25,    27,   29,   31,   33,   35,   37,   39,   41,  43,   45,    49,   53,   57,   61,   65,   69,   73,   77,   81,   87,   93,   99,  105,  111,  118]

def get_periodic_number(atom):
    try:
        return periodic_numbers[chemical_symbols.index(atom)]
    except IndexError:
        raise RuntimeError("Unknown atomic symbol: %s" % atom)
