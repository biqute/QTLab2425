import time

from Instruments.QuickSynE3615A import QuickSyn

print('Test QuickSynE3615A')

gen_1= QuickSyn('COM3')
gen_2= QuickSyn('COM7')

gen_1.set_frequency(5, 'GHz')
gen_2.set_frequency(3, 'GHz')
time.sleep(1)
result = gen_1.get_frequency('GHz')
print(f'Frequency: {result} GHz')
result_MHz = gen_1.get_frequency('MHz')
print(f'Frequency: {result_MHz} MHz')
#gen.close_conncetion()