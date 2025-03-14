import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time
from src.abstract.Instrument import Instrument
import threading

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
    
    def acq_streaming(self, sample_interval: int = 250, 
                    sample_units = ps.PS5000A_TIME_UNITS['PS5000A_US'], 
                    buffer_size: int = 500):
        """
        Start streaming data acquisition continuously until stop_streaming() is called.
        The method sets up a single buffer (for channel A) and registers a callback that appends
        each received chunk to self._streamed_data.
        """
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
        print(f"Streaming started at sample interval {sample_interval_ct.value * 1000} ns.")

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