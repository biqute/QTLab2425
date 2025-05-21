from src.instruments.PicoScope import PicoScope
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.ps5000a import ps5000a as ps
import h5py


if __name__ == '__main__':
    
    # Create an instance of PicoScope using 12-bit resolution 
    scope = PicoScope("12") 
    scope.initialize()
    
    print(scope.resolution)
    
    # Set up channel A as in the example:
    enabled = True
    analogue_offset = 0.0

    scope.set_channel('A', enabled, 'DC', '500MV', analogue_offset)
    
    #test send_TTL_trigger method
    scope.send_TTL_trigger(voltage = 3)
    
    time.sleep(0.5)
    
    scope.send_TTL_trigger(voltage = 2)
    
    time.sleep(0.5)
    
    scope.close_connection()
    
    
    
#-------------- old code ----------------    

    # # Create an instance of PicoScope using 12-bit resolution 
    # scope = PicoScope("12") 
    # scope.initialize()
    
    # print(scope.resolution)
    
    # # Set up channel A as in the example:
    # enabled = True
    # analogue_offset = 0.0

    # scope.set_channel('A', enabled, 'DC', '500MV', analogue_offset)
    # scope.set_channel('B', enabled, 'DC', '500MV', analogue_offset)
    # print("Channel A set up.")
    
    # # TEST SAMPLING INTERVAL AND TME BASE
    
    # # print('best timebase ' + str(scope.get_best_timebase()))
    
    # # time_base = scope.calculate_timebase(sampling_rate=50e6)
    # # print(f'Timebase: {time_base}')
    # # sampling_interval = scope.get_sampling_interval(time_base)
    # # print(f'Sampling interval: {sampling_interval}') 
    
    # # Trigger test
    # scope.set_trigger('A', 0, 'Rising', 0)
    # print("Trigger set up.")
    # time.sleep(1)
    # scope.disable_trigger()
    # print("Trigger disabled.")
       
    # # TEST DATA ACQUISITION BLOCK
    # data = scope.acq_block(250e6, post_trigger_samples=500, pre_trigger_samples=0, downsampling_ratio=2, downsampling_mode='none')
    # print("Data acquisition complete.")
    # print(len(data['A']))
    
    # # Plot the data
    # t = data['time']
    # data_A = data['A']
    # data_B = data['B']
    
    # # Save data to hdf5
    # with h5py.File('test_data.h5', 'w') as f:
    #     # 3D Dataset for t, data_A, data_B
    #     f.create_dataset('data', data=np.array([t, data_A, data_B]), compression="gzip")
    #     # Attributes for metadata
    #     f['data'].attrs['sampling_rate'] = 250e6
    #     f['data'].attrs['post_trigger_samples'] = 500
    #     f['data'].attrs['pre_trigger_samples'] = 0
    #     f['data'].attrs['downsampling_ratio'] = 2
    #     f['data'].attrs['downsampling_mode'] = 'none'
    #     f['data'].attrs['trigger_channel'] = 'A'
    #     f['data'].attrs['trigger_level'] = 0
    #     f['data'].attrs['trigger_slope'] = 'Rising'
    #     f['data'].attrs['trigger_delay'] = 0
    #     f['data'].attrs['column_names'] = ['time', 'I', 'Q']
    #     f['data'].attrs['column_units'] = ['s', 'mV', 'mV']
    #     f['data'].attrs['description'] = 'PicoScope data acquisition example'
    #     f['data'].attrs['Notes'] = 'Remember to attenuate the RF signal to avoid saturation.'
        
        
    # # Read the data from the hdf5 file
    # with h5py.File('test_data.h5', 'r') as f:
    #     data = f['data'][:]
    #     t = data[0]
    #     data_A = data[1]
    #     data_B = data[2]
    #     # Metadata
    #     column_names = f['data'].attrs['column_names']
    #     column_units = f['data'].attrs['column_units']
        
    # # Use the read data to do the plotting
    # # A is I and B is Q
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].plot(t, data_A, label=column_names[1], color='blue', linewidth=1, marker='o')
    # ax[0].plot(t, data_B, label=column_names[2], color='orange', linewidth=1, marker='o')
    # ax[0].legend()
    # ax[0].set_title('PicoScope Data Acquisition')
    # ax[0].set_xlabel(f'Time ({column_units[0]})')
    # ax[0].set_ylabel(f'Amplitude ({column_units[1]})')
    
    # # I-Q plot
    # ax[1].scatter(data_A, data_B, label='I-Q', color='green', s=10)
    # ax[1].set_title('I-Q Plot')
    # ax[1].set_xlabel(f'I ({column_units[1]})')
    # ax[1].set_ylabel(f'Q ({column_units[2]})')
    
    # ax[1].axis('equal')
    
    # fig.tight_layout()
    # plt.show()
    

    # # display status returns
    # print(scope.status)