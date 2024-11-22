import time

from Instruments.QuickSynE3615A import QuickSyn

print('Test QuickSynE3615A')

gen = QuickSyn('COM3')

gen.set_frequency(5, 'GHz')
time.sleep(1)
result = gen.get_frequency('GHz')
print(f'Frequency: {result} GHz')
result_MHz = gen.get_frequency('MHz')
print(f'Frequency: {result_MHz} MHz')
#gen.close_conncetion()