import os
import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.interpolation import InterpolationData
from euphonic.data.phonon import PhononData
from .utils import get_data_path


class TestReorderFreqsNaH(unittest.TestCase):

    def test_reorder_freqs(self):
        seedname = 'NaH-reorder-test'
        path = get_data_path()
        data = PhononData.from_castep(seedname, path=path)
        data.convert_e_units('1/cm')
        data.reorder_freqs()
        freqs = data.freqs.magnitude
        reordered_freqs = freqs[np.arange(len(freqs))[:, np.newaxis],
                                data._mode_map]
        expected_reordered_freqs = np.array(
            [[91.847109, 91.847109, 166.053018,
              564.508299, 564.508299, 884.068976],
             [154.825631, 132.031513, 206.21394,
              642.513551, 690.303338, 832.120011],
             [106.414367, 106.414367, 166.512415,
              621.498613, 621.498613, 861.71391],
             [-4.05580000e-02, -4.05580000e-02,
              1.23103200e+00, 5.30573108e+02,
              5.30573108e+02, 8.90673361e+02],
             [139.375186, 139.375186, 207.564309,
              686.675791, 686.675791, 833.291584],
             [123.623059, 152.926351, 196.644517,
              586.674239, 692.696132, 841.62725],
             [154.308477, 181.239973, 181.239973,
              688.50786, 761.918164, 761.918164],
             [124.976823, 124.976823, 238.903818,
              593.189877, 593.189877, 873.903056]])
        npt.assert_allclose(reordered_freqs, expected_reordered_freqs)


class TestReorderFreqsLZO(unittest.TestCase):

    def setUp(self):
        # Create both PhononData and InterpolationData objs for testing
        seedname = 'La2Zr2O7'
        data_path = get_data_path()
        self.pdata = PhononData.from_castep(seedname, path=data_path)
        self.pdata.convert_e_units('1/cm')
        ipath = os.path.join(data_path, 'interpolation', 'LZO')
        self.idata = InterpolationData.from_castep(seedname, path=ipath)
        self.idata.convert_e_units('1/cm')

        self.expected_reordered_freqs = np.array(
            [[65.062447, 65.062447, 70.408176, 76.847761, 76.847761,
              85.664054, 109.121893, 109.121893, 117.920003, 119.363588,
              128.637195, 128.637195, 155.905812, 155.905812, 160.906969,
              170.885818, 172.820917, 174.026075, 178.344487, 183.364621,
              183.364621, 199.25343, 199.25343, 222.992334, 225.274444,
              231.641854, 253.012884, 265.452117, 270.044891, 272.376357,
              272.376357, 275.75891, 299.890562, 299.890562, 315.067652,
              315.067652, 319.909059, 338.929562, 338.929562, 339.067304,
              340.308461, 349.793091, 376.784786, 391.288446, 391.288446,
              396.109935, 408.179774, 408.179774, 410.991152, 421.254131,
              456.215732, 456.215732, 503.360953, 532.789756, 532.789756,
              545.400861, 548.704226, 552.622463, 552.622463, 557.488238,
              560.761581, 560.761581, 618.721858, 734.650232, 739.200593,
              739.200593],
             [62.001197, 62.001197, 67.432601, 70.911126, 70.911126,
              87.435181, 109.893289, 109.893289, 110.930712, 114.6143,
              129.226412, 129.226412, 150.593502, 150.593502, 148.065107,
              165.856823, 168.794942, 167.154743, 169.819174, 187.349434,
              187.349434, 202.003734, 202.003734, 221.329787, 231.797486,
              228.999412, 259.308314, 264.453017, 279.078288, 270.15176,
              270.15176, 278.861064, 300.349651, 300.349651, 311.929653,
              311.929653, 318.251662, 334.967743, 334.967743, 340.747776,
              338.357732, 356.048074, 372.658152, 395.526156, 395.526156,
              398.356528, 406.398552, 406.398552, 407.216469, 421.122741,
              460.527859, 460.527859, 486.346855, 533.694179, 533.694179,
              544.93361, 549.252501, 550.733812, 550.733812, 558.006939,
              559.641583, 559.641583, 591.170512, 739.589673, 738.563124,
              738.563124],
             [55.889266, 55.889266, 64.492348, 66.375741, 66.375741,
              88.940906, 109.388591, 109.388591, 100.956751, 109.379914,
              130.017598, 130.017598, 145.579207, 145.579207, 134.563651,
              161.166842, 164.427227, 159.401681, 161.563336, 190.735683,
              190.735683, 205.550607, 205.550607, 219.351563, 238.204625,
              226.878861, 265.686284, 263.148071, 287.722953, 267.983859,
              267.983859, 281.041577, 299.480498, 299.480498, 308.176127,
              308.176127, 318.101514, 332.930623, 332.930623, 344.002317,
              335.480119, 361.930637, 368.350971, 399.050499, 399.050499,
              399.241143, 404.639113, 404.639113, 400.809087, 420.335936,
              465.504468, 465.504468, 470.205579, 534.544778, 534.544778,
              544.501022, 549.755212, 548.80696, 548.80696, 556.193672,
              558.101279, 558.101279, 565.776342, 741.372005, 737.860626,
              737.860626],
             [46.935517, 46.935517, 61.690137, 63.177342, 63.177342,
              90.180632, 107.721223, 107.721223, 86.944159, 104.159787,
              130.879196, 130.879196, 141.295304, 141.295304, 122.536218,
              157.146893, 160.037586, 151.613374, 153.750028, 193.160653,
              193.160653, 209.882364, 209.882364, 215.936117, 244.178665,
              225.432553, 272.052764, 261.655838, 295.533954, 265.906764,
              265.906764, 282.006965, 295.142911, 295.142911, 307.16826,
              307.16826, 319.295877, 332.071847, 332.071847, 348.814514,
              332.065989, 367.152249, 364.288189, 400.773283, 400.773283,
              399.790407, 404.068253, 404.068253, 387.165977, 418.829125,
              470.716023, 470.716023, 460.278318, 535.223077, 535.223077,
              544.111882, 550.193478, 547.016352, 547.016352, 552.362689,
              556.261571, 556.261571, 543.678775, 740.965394, 737.162508,
              737.162508],
             [36.367201, 36.367201, 59.168434, 60.36167, 60.36167,
              91.154677, 105.37576, 105.37576, 68.755044, 99.446481,
              131.658334, 131.658334, 138.017877, 138.017877, 113.14576,
              153.975056, 156.016054, 144.576942, 146.47047, 194.581347,
              194.581347, 214.716315, 214.716315, 210.473211, 249.235088,
              224.769091, 278.102009, 260.171794, 302.032435, 263.72796,
              263.72796, 282.018114, 289.408098, 289.408098, 308.097577,
              308.097577, 321.241146, 331.659808, 331.659808, 353.492915,
              328.675778, 371.468173, 362.406897, 399.901709, 399.901709,
              399.179346, 405.625572, 405.625572, 368.236337, 416.52493,
              475.665346, 475.665346, 458.944007, 535.667484, 535.667484,
              543.78033, 550.551048, 545.494533, 545.494533, 547.179463,
              554.338811, 554.338811, 524.846465, 739.380608, 736.536495,
              736.536495],
             [24.785718, 24.785718, 57.117299, 57.830885, 57.830885,
              91.859898, 103.047316, 103.047316, 47.456331, 95.691927,
              132.248074, 132.248074, 135.79383, 135.79383, 106.389552,
              151.718169, 152.772977, 138.984268, 139.88209, 195.244028,
              195.244028, 219.466615, 219.466615, 203.707835, 252.993107,
              224.615517, 283.248783, 258.912028, 306.841458, 261.246129,
              261.246129, 281.584343, 284.696598, 284.696598, 309.37963,
              309.37963, 323.205545, 331.373295, 331.373295, 353.088149,
              326.000428, 374.686778, 367.331006, 398.738183, 398.738183,
              398.433921, 407.157219, 407.157219, 349.637392, 413.438689,
              479.806857, 479.806857, 463.608166, 535.889622, 535.889622,
              543.524255, 550.815232, 544.325882, 544.325882, 541.757933,
              552.630089, 552.630089, 508.677347, 737.533584, 736.042236,
              736.042236],
             [12.555025, 12.555025, 55.757043, 55.972359, 55.972359,
              92.288749, 101.380298, 101.380298, 24.214202, 93.270077,
              132.593517, 132.593517, 134.540163, 134.540163, 102.211134,
              150.378051, 150.665566, 135.38769, 134.747421, 195.473725,
              195.473725, 223.210107, 223.210107, 197.85154, 255.276828,
              224.659659, 286.758828, 258.068085, 309.776254, 258.846604,
              258.846604, 281.147316, 281.874849, 281.874849, 310.385381,
              310.385381, 324.609898, 331.158402, 331.158402, 351.072968,
              324.818103, 376.67194, 374.186388, 397.950964, 397.950964,
              397.878833, 408.114477, 408.114477, 336.37863, 410.112489,
              482.591747, 482.591747, 471.735469, 535.964134, 535.964134,
              543.361599, 550.977172, 543.571634, 543.571634, 537.566668,
              551.451065, 551.451065, 494.626062, 736.133131, 735.72609,
              735.72609],
             [-0.019621, -0.019621, 55.277927, 55.277927, 55.277927,
              92.432911, 100.780857, 100.780857, -0.019621, 92.432911,
              132.696363, 132.696363, 134.147102, 134.147102, 100.780857,
              149.934817, 149.934817, 134.147102, 132.696363, 195.519690,
              195.519690, 224.698049, 224.698049, 195.519690, 256.039866,
              224.698049, 288.011070, 257.771213, 310.763767, 257.771213,
              257.771213, 280.972846, 280.972846, 280.972846, 310.763767,
              310.763767, 325.114540, 331.073494, 331.073494, 350.234619,
              325.114540, 377.342620, 377.342620, 397.677533, 397.677533,
              397.677533, 408.435923, 408.435923, 331.073494, 408.435923,
              483.578389, 483.578389, 480.948578, 535.976810, 535.976810,
              543.305729, 551.031712, 543.305729, 543.305729, 535.976810,
              551.031712, 551.031712, 483.578389, 735.617369,
              735.617369, 735.617369],
             [12.555025, 12.555025, 55.757043, 55.972359, 55.972359,
              92.288749, 101.380298, 101.380298, 24.214202, 93.270077,
              132.593517, 132.593517, 134.540163, 134.540163, 102.211134,
              150.378051, 150.665566, 135.38769, 134.747421, 195.473725,
              195.473725, 223.210107, 223.210107, 197.85154, 255.276828,
              224.659659, 286.758828, 258.068085, 309.776254, 258.846604,
              258.846604, 281.147316, 281.874849, 281.874849, 310.385381,
              310.385381, 324.609898, 331.158402, 331.158402, 351.072968,
              324.818103, 376.67194, 374.186388, 397.950964, 397.950964,
              397.878833, 408.114477, 408.114477, 336.37863, 410.112489,
              482.591747, 482.591747, 471.735469, 535.964134, 535.964134,
              543.361599, 550.977172, 543.571634, 543.571634, 537.566668,
              551.451065, 551.451065, 494.626062, 736.133131, 735.72609,
              735.72609],
             [24.785718, 24.785718, 57.117299, 57.830885, 57.830885,
              91.859898, 103.047316, 103.047316, 47.456331, 95.691927,
              132.248074, 132.248074, 135.79383, 135.79383, 106.389552,
              151.718169, 152.772977, 138.984268, 139.88209, 195.244028,
              195.244028, 219.466615, 219.466615, 203.707835, 252.993107,
              224.615517, 283.248783, 258.912028, 306.841458, 261.246129,
              261.246129, 281.584343, 284.696598, 284.696598, 309.37963,
              309.37963, 323.205545, 331.373295, 331.373295, 353.088149,
              326.000428, 374.686778, 367.331006, 398.738183, 398.738183,
              398.433921, 407.157219, 407.157219, 349.637392, 413.438689,
              479.806857, 479.806857, 463.608166, 535.889622, 535.889622,
              543.524255, 550.815232, 544.325882, 544.325882, 541.757933,
              552.630089, 552.630089, 508.677347, 737.533584, 736.042236,
              736.042236]])

    def test_reorder_freqs_phonon_data(self):
        self.pdata.reorder_freqs()
        freqs = self.pdata.freqs.magnitude
        reordered_freqs = freqs[np.arange(len(freqs))[:, np.newaxis],
                                self.pdata._mode_map]
        npt.assert_allclose(reordered_freqs,
                            self.expected_reordered_freqs)

    def test_reorder_freqs_interpolation_data(self):
        self.idata.calculate_fine_phonons(self.pdata.qpts, asr='realspace')
        self.idata.reorder_freqs()
        freqs = self.idata.freqs.magnitude
        reordered_freqs = freqs[np.arange(len(freqs))[:, np.newaxis],
                                self.idata._mode_map]
        # Set atol = 0.02 as this is the max difference between the
        # interpolated phonon freqs and phonons read from .phonon file
        npt.assert_allclose(reordered_freqs,
                            self.expected_reordered_freqs, atol=0.02)

    def test_reorder_freqs_interpolation_data_c(self):
        self.idata.calculate_fine_phonons(self.pdata.qpts, asr='realspace',
                                          use_c=True, fall_back_on_python=False)
        self.idata.reorder_freqs()
        freqs = self.idata.freqs.magnitude
        reordered_freqs = freqs[np.arange(len(freqs))[:, np.newaxis],
                                self.idata._mode_map]
        # Set atol = 0.02 as this is the max difference between the
        # interpolated phonon freqs and phonons read from .phonon file
        npt.assert_allclose(reordered_freqs,
                            self.expected_reordered_freqs, atol=0.02)

    def test_reorder_freqs_interpolation_data_c_2threads(self):
        self.idata.calculate_fine_phonons(self.pdata.qpts, asr='realspace',
                                          use_c=True, n_threads=2, fall_back_on_python=False)
        self.idata.reorder_freqs()
        freqs = self.idata.freqs.magnitude
        reordered_freqs = freqs[np.arange(len(freqs))[:, np.newaxis],
                                self.idata._mode_map]
        # Set atol = 0.02 as this is the max difference between the
        # interpolated phonon freqs and phonons read from .phonon file
        npt.assert_allclose(reordered_freqs,
                            self.expected_reordered_freqs, atol=0.02)

    def test_empty_interpolation_data_raises_exception(self):
        # Test that trying to call reorder on an empty object raises Exception
        self.assertRaises(Exception, self.idata.reorder_freqs)

    def test_reorder_freqs_interpolation_data_reorder_gamma_false(self):
        expected_freqs_gamma_false = np.array(
            [[65.062132,  65.062132,  70.408169,  76.847265,  76.847265,
              85.664045, 109.125801, 109.125801, 117.919526, 119.364404,
              128.637703, 128.637703, 155.910455, 155.910455, 160.910871,
              170.885802, 172.822572, 174.026059, 178.345426, 183.367994,
              183.367994, 199.255095, 199.255095, 222.994511, 225.276957,
              231.643260, 253.013523, 265.452092, 270.047653, 272.380029,
              272.380029, 275.758885, 299.891656, 299.891656, 315.068235,
              315.068235, 319.909029, 338.929042, 338.929042, 339.067179,
              340.309690, 349.793058, 376.785410, 391.289550, 391.289550,
              396.109878, 408.180646, 408.180646, 410.991897, 421.255699,
              456.216972, 456.216972, 503.360645, 532.791178, 532.791178,
              545.400810, 548.704174, 552.622688, 552.622688, 557.489359,
              560.761823, 560.761823, 618.722406, 734.651161, 739.201493,
              739.201493],
             [61.999469,  61.999469,  67.432594,  70.910761,  70.910761,
              87.435173, 109.896794, 109.896794, 110.929559, 114.614953,
              129.226745, 129.226745, 150.597131, 150.597131, 148.069534,
              165.856808, 168.796307, 167.154727, 169.819229, 187.352082,
              187.352082, 202.005200, 202.005200, 221.332088, 231.799285,
              229.000171, 259.308848, 264.452992, 279.080661, 270.154637,
              270.154637, 278.861037, 300.351022, 300.351022, 311.929630,
              311.929630, 318.251632, 334.967538, 334.967538, 340.747430,
              338.358944, 356.048040, 372.658702, 395.527431, 395.527431,
              398.356255, 406.398850, 406.398850, 407.216842, 421.124052,
              460.528885, 460.528885, 486.346761, 533.695614, 533.695614,
              544.933558, 549.252449, 550.733983, 550.733983, 558.008277,
              559.641679, 559.641679, 591.170802, 739.590447, 738.563858,
              738.563858],
             [55.886060,  55.886060,  64.492341,  66.375973,  66.375973,
              88.940898, 109.391449, 109.391449, 100.955203, 109.380390,
              130.017812, 130.017812, 145.581718, 145.581718, 134.567868,
              161.166826, 164.428235, 159.401666, 161.562989, 190.737868,
              190.737868, 205.551536, 205.551536, 219.353994, 238.205738,
              226.878895, 265.686727, 263.148046, 287.724839, 267.986101,
              267.986101, 281.041550, 299.481709, 299.481709, 308.175725,
              308.175725, 318.101484, 332.930764, 332.930764, 344.001848,
              335.481253, 361.930603, 368.351434, 399.051669, 399.051669,
              399.241656, 404.639031, 404.639031, 400.808128, 420.336909,
              465.505274, 465.505274, 470.205741, 534.546083, 534.546083,
              544.500971, 549.755161, 548.807074, 548.807074, 556.195034,
              558.101251, 558.101251, 565.776425, 741.372596, 737.861174,
              737.861174],
             [46.931953,  46.931953,  61.690131,  63.177604,  63.177604,
              90.180623, 107.723283, 107.723283,  86.942872, 104.160096,
              130.879324, 130.879324, 141.296831, 141.296831, 122.539387,
              157.146878, 160.038242, 151.613360, 153.749595, 193.162390,
              193.162390, 209.882716, 209.882716, 215.938307, 244.179220,
              225.432208, 272.053111, 261.655813, 295.535304, 265.908497,
              265.908497, 282.006938, 295.143235, 295.143235, 307.168275,
              307.168275, 319.295847, 332.072124, 332.072124, 348.813998,
              332.066986, 367.152214, 364.288587, 400.773645, 400.773645,
              399.789775, 404.068529, 404.068529, 387.165814, 418.829742,
              470.716601, 470.716601, 460.278606, 535.224109, 535.224109,
              544.111830, 550.193426, 547.016419, 547.016419, 552.363872,
              556.261463, 556.261463, 543.678747, 740.965792, 737.162867,
              737.162867],
             [36.364001,  36.364001,  59.168428,  60.361816,  60.361816,
              91.154668, 105.377005, 105.377005,  68.754358,  99.446652,
              131.658390, 131.658390, 138.018681, 138.018681, 113.147567,
              153.975041, 156.016415, 144.576929, 146.470132, 194.582516,
              194.582516, 214.716300, 214.716300, 210.474740, 249.235285,
              224.768778, 278.102239, 260.171769, 302.033261, 263.729207,
              263.729207, 282.018088, 289.407971, 289.407971, 308.097728,
              308.097728, 321.241116, 331.660046, 331.660046, 353.492462,
              328.676578, 371.468138, 362.407202, 399.901530, 399.901530,
              399.178947, 405.626046, 405.626046, 368.235901, 416.525239,
              475.665697, 475.665697, 458.944214, 535.668156, 535.668156,
              543.780279, 550.550996, 545.494557, 545.494557, 547.180304,
              554.338680, 554.338680, 524.846405, 739.380824, 736.536682,
              736.536682],
             [24.783339,  24.783339,  57.117293,  57.830943,  57.830943,
              91.859889, 103.047884, 103.047884,  47.456079,  95.691998,
              132.248079, 132.248079, 135.794171, 135.794171, 106.390320,
              151.718155, 152.773127, 138.984255, 139.881911, 195.244608,
              195.244608, 219.466485, 219.466485, 203.708620, 252.993137,
              224.615342, 283.248888, 258.912003, 306.841840, 261.246836,
              261.246836, 281.584316, 284.696416, 284.696416, 309.379717,
              309.379717, 323.205515, 331.373417, 331.373417, 353.087999,
              326.000973, 374.686743, 367.331029, 398.738011, 398.738011,
              398.433711, 407.157466, 407.157466, 349.636948, 413.438786,
              479.807006, 479.807006, 463.608242, 535.889935, 535.889935,
              543.524204, 550.815181, 544.325867, 544.325867, 541.758362,
              552.629981, 552.629981, 508.677287, 737.533650, 736.042286,
              736.042286],
             [12.553773,  12.553773,  55.757038,  55.972369,  55.972369,
              92.288740, 101.380434, 101.380434,  24.214147,  93.270088,
              132.593503, 132.593503, 134.540242, 134.540242, 102.211314,
              150.378037, 150.665591, 135.387678, 134.747366, 195.473867,
              195.473867, 223.210030, 223.210030, 197.851743, 255.276810,
              224.659595, 286.758836, 258.068060, 309.776334, 258.846802,
              258.846802, 281.147289, 281.874766, 281.874766, 310.385385,
              310.385385, 324.609867, 331.158414, 331.158414, 351.072912,
              324.818337, 376.671905, 374.186356, 397.950883, 397.950883,
              397.878750, 408.114517, 408.114517, 336.378369, 410.112479,
              482.591753, 482.591753, 471.735460, 535.964182, 535.964182,
              543.361548, 550.977120, 543.571592, 543.571592, 537.566753,
              551.450995, 551.450995, 494.626007, 736.133097, 735.726052,
              735.726052],
             [-0.000275,  -0.000275,  55.277922,  55.277922,  55.277922,
              92.432903, 100.780848, 100.780848,  -0.000275,  92.432903,
              132.696350, 132.696350, 132.696350, 134.147089, 100.780848,
              149.934803, 149.934803, 134.147089, 134.147089, 195.519672,
              195.519672, 224.698028, 224.698028, 195.519672, 256.039842,
              224.698028, 288.011043, 257.771189, 310.763738, 257.771189,
              257.771189, 280.972819, 280.972819, 280.972819, 310.763738,
              310.763738, 325.114509, 331.073463, 331.073463, 350.234586,
              325.114509, 377.342585, 377.342585, 397.677495, 397.677495,
              397.677495, 408.435884, 408.435884, 331.073463, 408.435884,
              483.578344, 483.578344, 480.948533, 535.976760, 535.976760,
              543.305678, 551.031660, 543.305678, 543.305678, 535.976760,
              551.031660, 551.031660, 483.578344, 735.617300, 735.617300,
              735.617300],
             [12.553773,  12.553773,  55.757038,  55.972369,  55.972369,
              92.288740, 101.380434, 101.380434,  24.214147,  93.270088,
              132.593503, 132.593503, 134.540242, 134.540242, 102.211314,
              150.378037, 150.665591, 135.387678, 134.747366, 195.473867,
              195.473867, 223.210030, 223.210030, 197.851743, 255.276810,
              224.659595, 286.758836, 258.068060, 309.776334, 258.846802,
              258.846802, 281.147289, 281.874766, 281.874766, 310.385385,
              310.385385, 324.609867, 331.158414, 331.158414, 351.072912,
              324.818337, 376.671905, 374.186356, 397.950883, 397.950883,
              397.878750, 408.114517, 408.114517, 336.378369, 410.112479,
              482.591753, 482.591753, 471.735460, 535.964182, 535.964182,
              543.361548, 550.977120, 543.571592, 543.571592, 537.566753,
              551.450995, 551.450995, 494.626007, 736.133097, 735.726052,
              735.726052],
             [24.783339,  24.783339,  57.117293,  57.830943,  57.830943,
              91.859889, 103.047884, 103.047884,  47.456079,  95.691998,
              132.248079, 132.248079, 135.794171, 135.794171, 106.390320,
              151.718155, 152.773127, 138.984255, 139.881911, 195.244608,
              195.244608, 219.466485, 219.466485, 203.708620, 252.993137,
              224.615342, 283.248888, 258.912003, 306.841840, 261.246836,
              261.246836, 281.584316, 284.696416, 284.696416, 309.379717,
              309.379717, 323.205515, 331.373417, 331.373417, 353.087999,
              326.000973, 374.686743, 367.331029, 398.738011, 398.738011,
              398.433711, 407.157466, 407.157466, 349.636948, 413.438786,
              479.807006, 479.807006, 463.608242, 535.889935, 535.889935,
              543.524204, 550.815181, 544.325867, 544.325867, 541.758362,
              552.629981, 552.629981, 508.677287, 737.533650, 736.042286,
              736.042286]])
        self.idata.calculate_fine_phonons(self.pdata.qpts, asr='realspace')
        self.idata.reorder_freqs(reorder_gamma=False)
        freqs = self.idata.freqs.magnitude
        reordered_freqs = freqs[np.arange(len(freqs))[:, np.newaxis],
                                self.idata._mode_map]
        npt.assert_allclose(reordered_freqs,
                            expected_freqs_gamma_false, rtol=0.02)

