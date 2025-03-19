from src.instruments.PicoScope import PicoScope
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.ps5000a import ps5000a as ps


if __name__ == '__main__':
    # Create an instance of PicoScope using 12-bit resolution 
    scope = PicoScope("PS5000A_DR_12BIT") 
    scope.initialize()
    
    print(scope.resolution)
    
    # Set up channel A as in the example:
    enabled = 1
    analogue_offset = 0.0
    channel_range = ps.PS5000A_RANGE['PS5000A_2V']
    scope.status["setChA"] = ps.ps5000aSetChannel(scope.chandle,
                                                ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                                enabled,
                                                ps.PS5000A_COUPLING['PS5000A_DC'],
                                                channel_range,
                                                analogue_offset)
    assert_pico_ok(scope.status["setChA"])
    print("Channel A set up.")
    
    # TEST SAMPLING INTERVAL AND TME BASE
    
    print('best timebase ' + str(scope.get_best_timebase()))
    
    time_base = scope.calculate_timebase(3)
    print(f'Timebase: {time_base}')
    sampling_interval = scope.get_sampling_interval(time_base)
    print(f'Sampling interval: {sampling_interval}') 
    
       

    # Start streaming acquisition
    print("Starting streaming acquisition for 5 seconds...")
    
    sample_interval = 2000
    sample_units = scope.get_command_value("TIME_UNITS", "PS")
    scope.acq_streaming(sample_interval=sample_interval, sample_units=sample_units, buffer_size=500)
    

    # Let the scope stream data for 5 seconds
    # time.sleep(5e-7*sample_interval)
    time.sleep(1)

    # Stop streaming
    scope.stop_streaming()

    # Retrieve the streamed data
    streamed_data = scope.get_streamed_data()
    print(f"Number of samples acquired: {len(streamed_data)}")

    # Optionally, convert raw ADC counts to mV using the adc2mV function (requires max ADC value)
    maxADC = ctypes.c_int16()
    scope.status["maximumValue"] = ps.ps5000aMaximumValue(scope.chandle, ctypes.byref(maxADC))
    assert_pico_ok(scope.status["maximumValue"])
    # Convert ADC counts to mV for channel A data
    voltage_data = adc2mV(streamed_data, channel_range, maxADC)

    # Plot the streamed voltage data
    plt.plot(voltage_data)
    plt.xlabel("Sample index")
    plt.ylabel("Voltage (mV)")
    plt.title("Streamed Data from PicoScope (Channel A)")
    plt.show()

    # Clean up and close the scope
    scope.kill()
    print("PicoScope connection closed.")
