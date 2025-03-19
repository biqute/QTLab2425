import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time
from src.abstract.Instrument import Instrument
import threading
import os
import json

class PicoScope(Instrument):
    def __init__(self, resolution: str, serial: str = None, name: str = None):
        # Set temp name if not provided
        name = name if name else "PicoScope_no_ID"
        super().__init__(name)
        
        self.resolution = ps.PS5000A_DEVICE_RESOLUTION[resolution]
        self.serial = serial
        
        # Create chandle and status ready for use
        self.chandle = ctypes.c_int16()
        self.status = {}
        
        self.pico_strings = None
    
    def initialize(self):
         # Returns handle to chandle for use in future API functions
        self.status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(self.chandle), None, self.resolution) #handle: generated id of scope, serial: serial of the scope to connect to, resolution: resolution of the scope

        try:
            assert_pico_ok(self.status["openunit"])
        except: # PicoNotOkError:

            powerStatus = self.status["openunit"]

            if powerStatus == 286:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            elif powerStatus == 282:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            else:
                raise

            assert_pico_ok(self.status["changePowerSource"])
        
        self.device_id = self.chandle.value
        self.name = self.name if self.name else f"PicoScope_{self.device_id}"
        
    def _Instrument__activate(self): # Dummy method
        pass
    
    def reset(self):
        pass
    
    def close_connection(self):
        
        # Disconnect the scope
        # handle = chandle
        self.status["close"] = ps.ps5000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])
        pass
    
    def shutdown(self):
        self.kill()
        print("Unplug PicoScope")
        
        
    def kill(self):
        
        # Stop the scope
        self.status["stop"] = ps.ps5000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])
    
        self.close_connection()
    
    def info(self, verbose=False):
        
        # Display status returns
        print(self.status)
        
    def calculate_timebase(self, sampling_interval: float, bit_mode = None) -> int:
        """
        Calculate the oscilloscope timebase corresponding to a given sampling interval and bit mode.
        
        The formulas are the inverse of the ones defined in the documentation:
        
        8-bit mode:
            - For sampling_interval < 8: timebase = log2(sampling_interval * 1)
            - For sampling_interval >= 8: timebase = round(sampling_interval * 125e-3) + 2
            
        12-bit mode:
            - For sampling_interval < 16: timebase = log2(sampling_interval * 5e-1) + 1
              (minimum value is 1)
            - For sampling_interval >= 16: timebase = round(sampling_interval * 62.5e-3) + 3
            
        14-bit mode and 15-bit mode:
            - For sampling_interval <= 8: timebase = 3
            - For sampling_interval > 8: timebase = round(sampling_interval * 125e-3) + 2
            
        16-bit mode:
            - For sampling_interval <= 16: timebase = 4
            - For sampling_interval > 16: timebase = round(sampling_interval * 62.5e-3) + 3
        
        Parameters:
        sampling_interval : float
            The desired sampling interval in seconds.
        bit_mode : int
            The resolution mode (8, 12, 14, 15, or 16).
            
        Returns:
        int
            The calculated timebase.
            
        Raises:
        ValueError: If an unsupported bit mode is provided.
        """
        bit_mode = self.get_command_value('RESOLUTION_BITMODE_INT', str(self.resolution)) if bit_mode is None else bit_mode
 
        if bit_mode == 8:
            # 8-bit mode:
            if sampling_interval < 8:
                if sampling_interval < 1:
                    raise ValueError("Sampling interval must be at least 1 ns for 8-bit mode.")
                # Inverse: n = log2(sampling_interval)
                return int(round(np.log2(sampling_interval))) 
            else:
                # Inverse: n = sampling_interval * 1.25e-1 + 2
                return int(round(sampling_interval * 1.25e-1)) + 2

        elif bit_mode == 12:
            # 12-bit mode: available timebases are 1, 2, 3, then n>=4.
            if sampling_interval < 16:
                if sampling_interval < 2:
                    raise ValueError("Sampling interval must be at least 2 ns for 12-bit mode.")
                # Inverse: n = log2(sampling_interval * 5e-1) + 1
                dep =  int(round(np.log2(sampling_interval * 5e-1))) + 1
                return dep if dep > 0 else 1
            else:
                # Inverse: n = sampling_interval * 0.0625 + 3
                return int(round(sampling_interval * 6.25e-2)) + 3

        elif bit_mode in (14, 15):
            if sampling_interval < 8:
                raise ValueError("Sampling interval must be at least 8 ns for 14/15-bit mode.")
            # 14-bit and 15-bit mode: valid timebases start at 3.
            if sampling_interval <= 8:
                return 3
            else:
                # Inverse: n = sampling_interval * 0.125 + 2
                return int(round(sampling_interval * 0.125)) + 2

        elif bit_mode == 16:
            if sampling_interval < 16:
                raise ValueError("Sampling interval must be at least 16 ns for 16-bit mode.")
            # 16-bit mode: valid timebases start at 4.
            if sampling_interval <= 16:
                return 4
            else:
                # Inverse: n = sampling_interval * 0.0625 + 3
                return int(round(sampling_interval * 6.25e-2)) + 3

        else:
            raise ValueError(f"Unsupported bit mode: {bit_mode}")
        
    def get_sampling_interval(self, timebase: int) -> float:
        """
        Calculate the sampling interval corresponding to a given timebase.
        
        Use the picoscope library "ps5000aGetTimebase" function.
        """
        sampling_interval = ctypes.c_float()
        max_samples = ctypes.c_int32()
        self.status["SamplingInterval"] = ps.ps5000aGetTimebase2(self.chandle, timebase, 1000, ctypes.byref(sampling_interval), ctypes.byref(max_samples), 0)
        
        print(sampling_interval.value)
        
        if sampling_interval.value < 1:
            raise ValueError("Sampling interval should be at least 1 ns. Something went wrong. Try different timebase.")
        
        return sampling_interval.value
    
    def get_best_timebase(self) -> int:
        status_key = "BestTimebase"
        timebase = ctypes.c_uint32()
        sampling_interval = ctypes.c_double()
        self.status[status_key] = ps.ps5000aGetMinimumTimebaseStateless(self.chandle,
                                                        self.get_command_value('CHANNEL_FLAGS', 'A'),
                                                        ctypes.byref(timebase),
                                                        ctypes.byref(sampling_interval),
                                                        self.resolution)
        return timebase.value
    
    def acq_block(self, sample_interval: int, sample_units, num_samples: int, buffer_size: int = 500):
        pass
    
    def acq_streaming(self, sample_interval: int = 250, 
                    sample_units = ps.PS5000A_TIME_UNITS['PS5000A_NS'], 
                    buffer_size: int = 500):
        """
        Start streaming data acquisition continuously until stop_streaming() is called.
        The method sets up a single buffer (for channel A) and registers a callback that appends
        each received chunk to self._streamed_data.
        """
        sample_units = sample_units if isinstance(sample_units, int) else ps.PS5000A_TIME_UNITS[sample_units]
        self._streaming_stop = False
        self._streamed_data = []  # reset storage
        self._buffer_size = buffer_size
        # Create a buffer for channel A (for simplicity, one channel is used)
        self._bufferA = np.zeros(shape=buffer_size, dtype=np.int16)

        # Register the data buffer with the driver for channel A
        status_key = "setDataBufferA_stream"
        self.status[status_key] = ps.ps5000aSetDataBuffer(self.chandle,
                                                        ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                                        self._bufferA.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                        buffer_size,
                                                        0,
                                                        ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'])
        assert_pico_ok(self.status[status_key])

        # Streaming mode: no pretrigger, no autoStop (0 means continuous)
        maxPreTriggerSamples = 0
        autoStopOn = 0  # continuous streaming
        downsampleRatio = 1
        totalSamples = 0  # not used in continuous mode

        sample_interval_ct = ctypes.c_int32(sample_interval)
        status_key = "runStreaming_stream"
        self.status[status_key] = ps.ps5000aRunStreaming(self.chandle,
                                                        ctypes.byref(sample_interval_ct),
                                                        sample_units,
                                                        maxPreTriggerSamples,
                                                        totalSamples,
                                                        autoStopOn,
                                                        downsampleRatio,
                                                        ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'],
                                                        buffer_size)
        assert_pico_ok(self.status[status_key])
        print(f"Streaming started at sample interval {sample_interval_ct.value} ns.")

        # Define callback function which the driver will call when new data is available.
        def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
            if noOfSamples > 0:
                # Copy the available samples from the buffer slice and store
                chunk = self._bufferA[startIndex:startIndex+noOfSamples].copy()
                self._streamed_data.append(chunk)
            # The callback should return an integer (0)
            return 0

        # Convert Python callback to C function pointer
        cFuncPtr = ps.StreamingReadyType(streaming_callback)

        # Start a thread that continuously polls for streaming data until stopped.
        def streaming_thread():
            while not self._streaming_stop:
                self.status["getStreaming"] = ps.ps5000aGetStreamingLatestValues(self.chandle, cFuncPtr, None)
                time.sleep(0.01)
            # When stop is signaled, stop the scope streaming
            self.status["stop_stream"] = ps.ps5000aStop(self.chandle)
            assert_pico_ok(self.status["stop_stream"])
            print("Streaming acquisition stopped.")

        self._streaming_thread = threading.Thread(target=streaming_thread)
        self._streaming_thread.start()

    def stop_streaming(self):
        """
        Signal to stop streaming acquisition and wait for the streaming thread to finish.
        """
        self._streaming_stop = True
        if self._streaming_thread is not None:
            self._streaming_thread.join()

    def get_streamed_data(self):
        """
        Returns the acquired streaming data as a single concatenated numpy array.
        """
        if self._streamed_data:
            return np.concatenate(self._streamed_data)
        else:
            return np.array([], dtype=np.int16)
        
    def set_channel(self, channel, enabled, coupling, range, offset):
        """
        Set the channel configuration.
        """
        status_key = f"setCh{channel}"
        range = ps.PS5000A_RANGE[self.get_command_value('RANGE', range)]
        self.status[status_key] = ps.ps5000aSetChannel(self.chandle,
                                                    ps.PS5000A_CHANNEL[f'PS5000A_CHANNEL_{channel}'],
                                                    enabled,
                                                    ps.PS5000A_COUPLING['PS5000A_DC' if coupling == 'DC' else 'PS5000A_AC'],
                                                    range,
                                                    offset)
    
    def get_command_value(self, dict_name, key):
        if self.pico_strings is None:  
            # Read the json PicoStrings file
            python_file_dir = os.path.dirname(__file__)
            with open(os.path.join(python_file_dir, 'cfgs/PicoStrings.json')) as f:
                self.pico_strings = json.load(f)
                
        # Get the value
        return self.pico_strings[dict_name][key]
