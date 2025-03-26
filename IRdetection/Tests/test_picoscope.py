from src.instruments.PicoScope import PicoScope
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.ps5000a import ps5000a as ps


if __name__ == '__main__':
    # Create an instance of PicoScope using 12-bit resolution 
    scope = PicoScope("PS5000A_DR_8BIT") 
    scope.initialize()
    
    print(scope.resolution)
    
    # Set up channel A as in the example:
    enabled = True
    analogue_offset = 0.0

    scope.set_channel('A', enabled, 'DC', '50MV', analogue_offset)
    print("Channel A set up.")
    
    # TEST SAMPLING INTERVAL AND TME BASE
    
    print('best timebase ' + str(scope.get_best_timebase()))
    
    time_base = scope.calculate_timebase(sampling_rate=50e6)
    print(f'Timebase: {time_base}')
    sampling_interval = scope.get_sampling_interval(time_base)
    print(f'Sampling interval: {sampling_interval}') 
    
    # Trigger test
    scope.set_trigger('A', 0, 'Rising', 0)
    print("Trigger set up.")
    time.sleep(1)
    scope.disable_trigger()
    print("Trigger disabled.")
       

    # # Stop the scope
    # # handle = chandle
    # scope.status["stop"] = ps.ps5000aStop(scope.chandle)
    # assert_pico_ok(scope.status["stop"])

    # # Close unit Disconnect the scope
    # # handle = chandle
    # scope.status["close"]=ps.ps5000aCloseUnit(scope.chandle)
    # assert_pico_ok(scope.status["close"])

    # display status returns
    print(scope.status)