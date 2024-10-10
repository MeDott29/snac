
from openai import OpenAI
from os import getenv

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "https://www.snacgenerator.io",  # Optional, for including your app on openrouter.ai rankings.
    "X-Title": "textToSnac",  # Optional. Shows in rankings on openrouter.ai.
  },
  model="liquid/lfm-40b",
  messages=[
    # {
    #   "role": "system",
    #   "content": "you must generate codes in the demonstrated format eg. [[],[],[],[]]."
    # },    
    {
      "role": "user",
      "content": """[{"freq": 585.2948802512542, "waveform_type": "triangle", "amplitude": 0.5702909940700135, "duration": 5}, {"freq": 321.1634172321212, "waveform_type": "sine", "amplitude": 0.7447116768063708, "duration": 5}]"""
    },
    {
      "role": "assistant",
      "content": """[[542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 542, 3062]], [[1017, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 1017, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 1890, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 1890, 3422, 1890, 3422, 3751, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 1890, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3725, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 1890, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 1890, 3422, 1890, 3422, 1528, 1890, 3422, 1890, 3422, 3422, 1890, 3422, 1890, 1890, 3422, 1471, 78]], [[2753, 1152, 2085, 656, 1498, 1365, 1024, 122, 1211, 2429, 3725, 1024, 551, 3951, 1902, 3725, 1420, 1162, 3951, 1299, 3725, 2028, 2446, 1876, 3854, 2744, 2028, 2446, 3654, 3609, 2744, 2028, 2446, 3654, 3609, 11, 2028, 1431, 3654, 3062, 11, 1230, 52, 3654, 3062, 1365, 2856, 3762, 3654, 2949, 11, 3854, 3657, 3654, 28, 11, 2828, 1240, 921, 142, 11, 2828, 1240, 3699, 1162, 1035, 3854, 1240, 1695, 1162, 330, 3609, 1240, 2105, 4019, 1732, 3609, 11, 233, 99, 1732, 3609, 11, 2105, 99, 3919, 3784, 11, 1323, 3434, 3919, 1274, 67, 2856, 3749, 3692, 2818, 2429, 939, 1152, 3692, 2818, 67, 1098, 921, 3692, 2664, 1035, 1098, 1152, 661, 1540, 229, 1098, 1306, 3160, 1876, 1521, 1098, 921, 22, 2071, 1963, 2744, 1479, 4079, 1668, 1263, 1806, 1479, 2630, 1668, 1222, 1806, 1960, 1960, 3749, 2006, 656, 2429, 1960, 1152, 2085, 656, 2429, 1365, 1306, 122, 1211, 2429, 3725, 1024, 2284, 3951, 1902, 3725, 1420, 1162, 3951, 244, 3725, 1420, 1162, 3263, 3854, 477, 2028, 2446, 3654, 3609, 2744, 2028, 2446, 3654, 3609, 11, 2028, 2078, 3654, 3062, 11, 1230, 1189, 3654, 3062, 1365, 2856, 1068, 3654, 2949, 11, 3854, 3657, 3654, 1059, 11, 1299, 1240, 3654, 142, 11, 2828, 1240, 3583, 142, 3951, 3854, 1240, 1695, 1162, 2715, 3854, 1240, 1695, 1162, 426, 3792, 11, 2105, 99, 1732, 3609, 11, 2105, 99, 3692, 3792, 11, 233, 729, 3692, 1274, 67, 2856, 3749, 3692, 2818, 2429, 3854, 1152, 3692, 2818, 67, 1098, 1306, 3692, 128, 1035, 1098, 1152, 123, 1540, 229, 1098, 1306, 1993, 3263, 757, 1098, 921, 2630, 2071, 1521, 3609, 1479, 4079, 1668, 1263, 3792, 1479, 2630, 1668, 1222, 1806, 1479, 1812, 3749, 2085, 3784, 3205, 1960, 2922, 2085, 1806, 2429, 1738, 2823, 3763]], [[3096, 2416, 143, 1230, 4079, 3517, 1667, 975, 2871, 758, 559, 532, 376, 2829, 1179, 3517, 900, 3980, 2489, 758, 2143, 3552, 1402, 2727, 2439, 3698, 2033, 3983, 1711, 3607, 2947, 2021, 958, 2727, 1535, 55, 2033, 1015, 821, 2641, 2947, 2021, 958, 2727, 32, 55, 747, 312, 758, 3577, 850, 975, 274, 2646, 3530, 1136, 2707, 3789, 3068, 826, 50, 975, 274, 1711, 2100, 643, 2269, 3789, 728, 3791, 2996, 970, 274, 3231, 2100, 3660, 1231, 1096, 3517, 25, 1057, 3983, 274, 2046, 3577, 517, 3789, 2727, 3517, 25, 2947, 2871, 3841, 43, 3552, 3348, 3789, 4079, 3517, 3660, 3980, 2871, 758, 559, 1685, 3348, 1096, 3042, 2988, 3514, 430, 2489, 758, 826, 3552, 110, 1402, 1535, 2988, 3698, 3599, 2489, 3607, 3953, 202, 110, 3120, 288, 2988, 464, 2605, 1909, 3607, 3953, 2943, 2021, 3120, 288, 3161, 2088, 3789, 2181, 821, 4048, 975, 3878, 1711, 288, 2694, 2088, 747, 728, 705, 1057, 975, 274, 1711, 2100, 3152, 3469, 747, 886, 2590, 2947, 230, 3841, 43, 2100, 1356, 1231, 1096, 886, 3161, 2947, 3910, 3841, 3607, 1190, 1617, 1231, 2787, 2952, 3517, 1550, 2871, 2181, 3607, 2143, 3469, 1122, 1711, 2988, 3517, 850, 975, 2265, 439, 4009, 2996, 1402, 3370, 2988, 2992, 3660, 2489, 2265, 826, 4009, 2996, 1402, 3068, 3530, 464, 1231, 2489, 1711, 758, 1818, 2021, 1402, 3370, 1535, 1706, 2033, 2181, 1711, 2641, 2947, 3878, 3120, 3025, 2988, 55, 2033, 2181, 826, 2143, 975, 274, 300, 3025, 3161, 3469, 747, 3789, 826, 3577, 50, 3841, 3942, 2646, 3152, 3469, 2707, 3454, 32, 3323, 542, 4089, 3942, 1190, 1137, 643, 2269, 886, 941, 3791, 1140, 3983, 43, 559, 2100, 3683, 3120, 1096, 3517, 2947, 1140, 3049, 439, 559, 216, 143, 3120, 4079, 3517, 1667, 975, 3049, 3394, 559, 532, 376, 2829, 4079, 3517, 3660, 3980, 2489, 1909, 3966, 3552, 3348, 2727, 1535, 2988, 2033, 3983, 1711, 3607, 2143, 3552, 958, 2727, 1535, 55, 2033, 1015, 2068, 3607, 2947, 2021, 700, 2727, 32, 55, 3313, 2426, 758, 826, 430, 975, 3878, 3120, 3530, 1136, 2707, 3789, 3068, 826, 3642, 975, 274, 1711, 2100, 1136, 2269, 3789, 728, 3791, 2996, 970, 274, 3231, 2100, 643, 1231, 1096, 886, 3161, 1057, 3983, 274, 2046, 3323, 1117, 2549, 2727, 3517, 25, 2947, 3910, 3841, 43, 1668, 1231, 3789, 4079, 3517, 2532, 3980, 2871, 758, 559, 4009, 1231, 398, 2646, 2988, 3514, 430, 2489, 2265, 826, 3552, 2996, 1402, 1535, 2988, 2283, 1015, 2489, 3607, 826, 3552, 110, 2424, 3530, 2988, 3698, 2033, 2489, 1711, 3953, 202, 2021, 3120, 288, 3577, 2088, 2494, 1909, 826, 3953, 2943, 1855, 1711, 2727, 3161, 2088, 747, 886, 705, 1136, 975, 274, 3393, 2100, 3152, 3469, 747, 886, 826, 2947, 230, 3841, 43, 2646, 3152, 3469, 1096, 886, 32, 2947, 3910, 3841, 43, 1190, 1617, 1231, 3847, 3913, 3517, 2947, 2871, 2181, 43, 559, 3469, 1231, 1230, 4079, 3517, 1550, 975, 2265, 3607, 4009, 2996, 1402, 1230, 3530, 2992, 3660, 975, 2265, 826, 4009, 2347, 376, 2522, 3530, 464, 2033, 2, 1711, 758, 1818, 2347, 1402, 728, 2754, 1706, 2033, 2489, 1711, 758, 2947, 3878, 958, 2727, 3577, 55, 2033, 2181, 826, 2143, 2947, 3878, 300, 3025, 3041, 3469, 3376, 858, 705, 3323, 1152, 975, 3942, 3120, 3152, 3469, 2707, 3454, 3383, 3791, 542, 975, 3942, 1711, 1137, 643, 2862, 886, 941, 3791, 1140, 3983, 274, 559, 3228, 3683, 3120, 1096, 941, 2947, 1140, 2871, 3607, 559, 216, 143, 3120, 4079, 3041, 1667, 2620, 2871, 758, 1538, 2067, 2916, 3616, 1332, 2817]]"""
    },
    {
      "role": "user",
      "content": """[{"freq": 368.0444764705185, "waveform_type": "triangle", "amplitude": 0.6256664311345882, "duration": 5}, {"freq": 363.75831332180394, "waveform_type": "square", "amplitude": 0.8250475701520514, "duration": 5}, {"freq": 785.6572871557975, "waveform_type": "sine", "amplitude": 0.7703993035281365, "duration": 5}]"""
    },
    {
      "role": "assistant",
      "content": """"[[[1602, 1062, 1602, 1602, 1062, 1602, 1602, 1062, 1602, 1602, 1602, 1602, 1602, 1602, 1062, 1602, 1602, 1062, 1602, 1602, 1602, 1602, 1602, 1602, 1062, 1602, 1602, 1062, 1602, 1602, 1602, 1062, 1602, 1602, 1062, 1602, 1602, 1062, 1602, 1602, 1602, 1602, 1602, 1602, 1062, 1602, 1602, 1602, 1602, 1602, 1602, 1602, 1602, 1602, 1062, 1602, 1602, 1062, 1602, 1602, 1602, 1062, 1602, 1602, 1062, 1602, 1602, 1602, 1602, 1602, 1602, 2945]], [[1949, 1949, 3863, 3257, 3447, 3830, 3331, 1949, 1779, 3257, 3987, 3830, 1114, 1949, 3331, 2171, 1949, 3987, 3447, 3987, 1949, 2171, 1949, 1949, 3987, 3447, 3987, 1949, 1779, 3863, 3987, 3447, 1114, 3987, 3863, 1779, 1949, 3987, 3447, 3987, 1949, 594, 1949, 1949, 3987, 3447, 3331, 1949, 507, 1949, 3987, 1114, 1319, 3987, 3863, 1779, 1949, 3331, 3830, 3987, 1949, 1949, 507, 3331, 3987, 3447, 3331, 1949, 2171, 1191, 3987, 3830, 1114, 3987, 1949, 1779, 1949, 3331, 3447, 3987, 1949, 1949, 1949, 1949, 3987, 3447, 3987, 1949, 1779, 3863, 1949, 3447, 3447, 3987, 1949, 1949, 1949, 3987, 3447, 1255, 1949, 1949, 1779, 1949, 3987, 3447, 1319, 1949, 507, 1949, 1949, 1114, 3830, 3987, 3863, 1779, 1949, 1949, 3830, 3447, 1949, 1949, 507, 3331, 3987, 3447, 1319, 1949, 507, 3331, 3331, 3987, 3830, 3987, 1949, 920, 1949, 1949, 1114, 3447, 1949, 1949, 332, 3401]], [[4002, 3072, 2372, 953, 3699, 1436, 3724, 2390, 316, 1, 2546, 2984, 316, 2028, 622, 3741, 3699, 2969, 3741, 1459, 3207, 586, 2035, 59, 800, 2028, 1779, 1, 3536, 3089, 3873, 1695, 664, 535, 2096, 3946, 1318, 3536, 833, 535, 3536, 3410, 2713, 4055, 3854, 239, 2462, 3719, 1850, 1031, 2828, 1850, 3189, 3487, 2684, 1850, 3487, 1385, 569, 1974, 2594, 569, 1974, 709, 1019, 3487, 3536, 1850, 1974, 2602, 569, 1974, 2713, 1823, 1277, 1093, 2897, 3487, 1, 1850, 696, 2218, 569, 1974, 2713, 3886, 3107, 2713, 4064, 3095, 2713, 569, 107, 2218, 569, 542, 2713, 2417, 2053, 724, 2083, 1738, 2713, 2305, 3774, 724, 2462, 2567, 2710, 2462, 1953, 724, 2305, 439, 3072, 2654, 3779, 83, 1695, 1436, 3072, 3857, 1436, 3072, 2390, 816, 1850, 3254, 800, 1850, 819, 2485, 3536, 2539, 1831, 1783, 3254, 2331, 3382, 1779, 586, 1783, 2587, 1031, 3741, 819, 2000, 3741, 3254, 2021, 1093, 2102, 2228, 3189, 2623, 648, 3189, 3303, 3740, 1093, 2105, 3571, 2713, 3487, 4076, 1850, 3487, 1466, 1850, 1831, 1498, 1850, 2462, 696, 1850, 2462, 1974, 1850, 2470, 1479, 1850, 36, 1783, 2866, 569, 3514, 1850, 569, 3514, 1850, 3886, 1783, 1850, 2107, 1783, 1850, 2718, 3741, 1401, 1333, 2640, 724, 3410, 3657, 4055, 2828, 2602, 3189, 2650, 2713, 3189, 1017, 2413, 3382, 3716, 3019, 2713, 2996, 2710, 3382, 1953, 3072, 2713, 1953, 2609, 2713, 2372, 3540, 724, 2372, 3216, 1850, 2083, 1385, 2390, 2780, 4055, 2546, 4050, 2130, 3420, 2305, 2130, 2028, 3487, 3724, 2390, 2941, 586, 2105, 1523, 1, 1852, 3886, 2331, 2028, 3266, 1015, 2105, 2647, 1, 2035, 2647, 1339, 1785, 819, 1339, 4055, 1619, 535, 2462, 3016, 724, 2096, 3534, 724, 2305, 1098, 709, 3189, 137, 648, 3189, 4002, 332, 86, 3362, 2418]], [[2727, 2004, 2083, 581, 1828, 3370, 2727, 3225, 2083, 2430, 2549, 1127, 2984, 2883, 1758, 768, 2549, 1127, 69, 2883, 261, 768, 2780, 2556, 515, 1789, 1483, 2933, 3439, 3606, 3902, 1789, 2083, 3108, 2217, 2611, 69, 589, 1758, 333, 1309, 3285, 1999, 1189, 295, 735, 87, 349, 724, 1557, 1226, 1523, 2943, 2366, 3902, 1557, 3920, 3727, 3141, 1215, 605, 2390, 295, 2701, 2081, 466, 1774, 1189, 2778, 2701, 3148, 2184, 1774, 2965, 1559, 1395, 1267, 535, 957, 2030, 1189, 3727, 2828, 466, 51, 1147, 168, 2701, 2583, 3797, 3088, 3920, 581, 2098, 4016, 1284, 3088, 3920, 581, 19, 2898, 3797, 51, 1147, 670, 117, 321, 2727, 51, 1195, 168, 1013, 3362, 3073, 3088, 1147, 168, 3692, 1784, 2239, 1875, 3168, 1716, 22, 2362, 2239, 1600, 1147, 3859, 1013, 1370, 3120, 2688, 1195, 2376, 1013, 3897, 3073, 2688, 1147, 2376, 606, 3181, 3480, 20, 1147, 782, 22, 1993, 3480, 20, 3836, 2668, 1152, 2822, 240, 2237, 2237, 3858, 1013, 4082, 3073, 1497, 1147, 3858, 1013, 2060, 47, 2551, 1147, 2704, 3300, 2759, 3480, 2648, 557, 3332, 1451, 3201, 47, 4019, 3305, 4075, 2011, 1606, 3789, 1789, 3836, 1665, 1583, 3437, 2677, 1789, 307, 2607, 2765, 1010, 1450, 2390, 3168, 2526, 1508, 384, 2406, 2390, 1883, 3563, 971, 2457, 886, 3340, 679, 2607, 114, 2175, 4045, 1955, 2083, 2607, 1723, 3606, 466, 2656, 2083, 1052, 1414, 1127, 69, 2656, 2796, 333, 1023, 1127, 1500, 1955, 1053, 3855, 1419, 3606, 3902, 1955, 2215, 581, 328, 3606, 3902, 2883, 2083, 581, 2217, 2104, 1500, 2883, 2083, 3108, 3750, 1127, 69, 2883, 2589, 768, 2217, 3606, 2727, 502, 1483, 1765, 3469, 4055, 724, 1789, 2132, 1765, 2217, 4055, 2237, 589, 295, 2506, 112, 466, 2934, 1189, 2908, 2430, 3683, 1215, 1774, 589, 3099, 2083, 3140, 2, 2967, 2030, 3099, 3284, 87, 364, 2967, 589, 295, 1013, 1653, 2566, 1774, 1189, 581, 2701, 93, 2967, 2814, 117, 581, 2098, 624, 2181, 957, 2390, 1352, 3284, 2695, 1948, 102, 2390, 117, 2701, 3222, 3902, 3088, 1147, 581, 3694, 4016, 2967, 3088, 3168, 581, 19, 2270, 2239, 3088, 1147, 670, 2486, 1248, 1092, 51, 187, 1072, 2486, 3793, 2967, 3088, 2115, 2376, 3692, 154, 2237, 1875, 2115, 1716, 22, 136, 809, 1875, 2115, 3960, 4037, 1480, 2008, 1497, 1500, 3461, 1013, 2170, 2008, 1497, 3836, 3858, 3108, 2050, 809, 227, 3836, 782, 2217, 1197, 2329, 20, 307, 782, 3300, 2759, 3211, 1557, 3836, 2131, 2238, 1719, 2008, 1497, 3836, 2132, 2238, 1719, 2348, 2030, 3836, 2704, 1149, 1010, 1737, 2030, 2578, 677, 2269, 3201, 1737, 2243, 305, 677, 2011, 1099, 47, 3480, 1147, 2607, 209, 1099, 47, 1789, 3168, 2607, 2765, 384, 466, 2390, 3168, 3284, 1508, 3688, 1737, 3037, 1758, 1419, 958, 806, 466, 658, 1651, 3909, 277, 2981, 398, 1955, 2083, 1352, 328, 3370, 2967, 1955, 2083, 581, 958, 1127, 3667, 2883, 1758, 235, 2251, 1127, 1500, 3037, 2796, 768, 532, 3394, 3902, 1789, 2548, 735, 3385, 3606, 3902, 1789, 1038, 581, 2217, 1577, 1920, 2883, 2908, 581, 1653, 2611, 2934, 246, 2908, 3330, 2676, 705, 3173, 2390, 2548, 735, 2217, 349, 2239, 2237, 3099, 735, 2217, 1577, 724, 20, 295, 1013, 1653, 200, 1774, 589, 3727, 1637, 93, 200, 1854, 1189, 295, 3168, 1533, 3660, 724, 1076, 1189, 2236, 1309, 157, 51, 1147, 168, 1013, 2401, 466, 1774, 1899, 2778, 2701, 2034, 2967, 3088, 3202, 581, 2098, 2034, 1500, 3088, 3263, 1189, 2098, 875, 1092, 3088, 1147, 1352, 1013, 780, 2967, 3088, 2913, 581, 2667, 1200, 3616, 3685, 3406]]]"""
    },
    {
      "role": "user",
      "content": """[{"freq": 833.9058690311292, "waveform_type": "triangle", "amplitude": 0.5692705711523316, "duration": 5}, {"freq": 449.3381911356998, "waveform_type": "sine", "amplitude": 0.9898941538533828, "duration": 5}, {"freq": 175.43498169702127, "waveform_type": "sine", "amplitude": 0.776955829197643, "duration": 5}, {"freq": 718.0096801412656, "waveform_type": "sine", "amplitude": 0.9244855315206003, "duration": 5}]} """
    },
  ]
)
print(completion.choices[0].message.content)
