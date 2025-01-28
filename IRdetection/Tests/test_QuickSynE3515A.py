import time

from Instruments.QuickSynE3615A import QuickSyn

print('Test QuickSynE3615A')

gen_1= QuickSyn('COM8')
gen_2= QuickSyn('COM7')


gen_1.set_frequency(100, 'MHz')
gen_2.set_frequency(0.1005, 'GHz')
time.sleep(1)
#
result = gen_1.get_frequency('MHz')
print(f'Frequency: {result} MHz')
result_MHz = gen_2.get_frequency('GHz')
print(f'Frequency: {result_MHz} GHz')

# close connection
gen_1.close_connection()
gen_2.close_connection()
