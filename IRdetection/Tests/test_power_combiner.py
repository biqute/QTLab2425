import time
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, r'\Users\kid\labQT\Lab2024\SINGLE_PHOTON\QTLab2425\IRdetection')

from Instruments.QuickSynE3615A import QuickSyn
from Instruments.FSV3030 import FSV3030

print('Test Power Combiner')

#connection to instruments
gen_1= QuickSyn('COM3')
gen_2= QuickSyn('COM7')
psa = FSV3030('192.168.3.50')

#set frequencies
gen_1.set_frequency(1.5, 'GHz')
gen_1.set_output_state('ON')
gen_2.set_frequency(1.5, 'GHz')
gen_2.set_output_state('ON')
time.sleep(1)

#check setted frequencies
frequency_1 = gen_1.get_frequency('GHz')
print(f'Frequency first generator: {frequency_1} GHz')
frequency_2 = gen_2.get_frequency('GHz')
print(f'Frequency second generator: {frequency_2} GHz')

#set psa
frequency_center = (1.5)*1e9 + 1e3
print(f'Frequency center: {frequency_center} Hz')
frequency_span = 0.02*1e9
psa.set_resolution_bandwidth(100000)
psa.set_video_bandwidth_ratio(10)
psa.set_sweep_time("auto")
psa.set_sweep_type("FFT")
psa.set_sweep_count(10)        #number of averages 10
psa.set_sweep_points(5000)    #number of points 5000
psa.set_dB_reference(10)
psa.set_dB_range(100.0)
psa.set_dB_ref_position(95.0)
psa.set_trace_mode("AVERAGE")
psa.set_frequency_center(frequency_center)
psa.set_frequency_span(frequency_span)
psa.set_sweep_time('auto')



x_data, y_data = psa.get_spectrum()
print(f'Peak: {psa.get_peak(x_data, y_data)} dBm')

#12.73360062 - 5.120486259 = 7.613114361 -> 7.613114361 - 3.0 = 4.613114361

#write to file
f = open("power_combiner_runs/power_combiner_s1p1test.txt", "w") 
for i in range(len(x_data)):
    f.write(str(x_data[i]) + " " + str(y_data[i]) + "\n")
f.close()

#close connection
#gen_1.close_conncetion()
#gen_2.close_conncetion()
#psa.close_connection()