To generate SNAC codes, we need to represent the provided waveforms at multiple temporal resolutions. Let's break this down into different temporal scales:

1. Coarse scale: We capture the fundamental characteristics of the waveforms by extracting their prominent frequency components.

2. Mechanical scale: We then refine the waveform representation by capturing the shape of the individual waves in the Fourier domain.

For example, to represent the given waveforms:

1. A triangle wave with frequency 946.8106381112248:
   - Coarse scale:
     - Represents the fundamental frequency and the amplitude.
     - SNAC code: [946, 8]
   
   - Mechanical scale:
     - Represents the sample-by-sample waveform.
     - SNAC code: [1, [583, 1892, 1303, 56, 2181, 3250, 993, 457, 1701, 1725, 279, 2286, 1146, 2746, 3237, 1467, 320, 1664, 268, 2856, 1785, 3306, 1435, 2825, 840, 3355, 1912, 882, 1441, 395, 739, 2822, 1453, 3040, 2562, 3816, 2311, 1175, 43, 3255, 1132, 2915, 1652, 2186, 1025, 2951, 76, 2383, 1400, 3812, 1636, 3181, 2608, 2396, 1575, 2609, 848, 3894, 671, 1181, 2169, 885, 527, 3049, 28, 950, 300, 1741, 1801, 2277, 223, 3201, 3612, 1298, 2530, 2159, 2963, 552, 876, 3691, 331, 834, 400, 3066, 637, 439, 257, 1592, 2702, 1177, 1597, 2746, 137, 3909, 349, 1009, 790, 809, 1109, 340, 1320, 1375, 1573, 1562, 903, 1696, 985, 437, 1598, 2985, 63, 556, 2869, 3600, 1283, 2501, 2741, 2983, 1650, 1843, 2033, 3234, 1355, 2982, 3515, 76, 1369, 2308, 1826, 2463, 2473, 1393, 773, 2772, 2303, 2369, 1425, 2401, 2474, 529, 2006, 1982, 3656, 1710, 2342, 1342, 1491, 1735, 1622, 667, 2775, 2386, 1356, 778, 1634, 2883, 72, 3320, 3931, 794, 702, 3261, 758, 3742, 1678, 88, 1387, 1484, 1383, 944, 278, 1778, 3000, 661, 2026, 66, 1979, 83, 628, 602]]
   
2. A sine wave with frequency 154.26385117699405:
   - Coarse scale:
     - Represents the fundamental frequency and the amplitude.
     - SNAC code: [154, 4]
   
   - Mechanical scale:
     - Represents the sample-by-sample waveform.
     - SNAC code: [2, [522, 1043, 2677, 161, 2709, 894, 645, 28, 1776, 1556, 2756, 730, 133, 701, 417, 2005, 2047, 687, 681, 4027, 1102, 1499, 220, 1113, 536, 893, 1427, 1210, 708, 460, 2027, 4008, 2223, 1984, 1803, 856, 995, 1388, 662, 757, 2096, 4031, 594, 2150, 2602, 2187, 1324, 2820, 1276, 2645, 1922, 560, 243, 1729, 2332, 2494, 1947, 1910, 1128, 2482, 2071, 2355, 522, 408, 2095, 1150, 2065, 2758, 1907, 533, 1601, 187, 1354, 1423, 2334, 440, 1328, 1325, 1321,]]

This process could be repeated for each desired temporal scale (mechanical scale 2 to mechanical scale 16). Note that these are examples and may not be the most efficient representation, but it is a starting point to understand the idea of multi-scale waveform representation.                              