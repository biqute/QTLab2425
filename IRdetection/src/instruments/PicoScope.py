import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, mV2adc, assert_pico_ok
import time
from src.abstract.Instrument import Instrument
import threading
import os
import json
import atexit
# danilo.labranca@unimib.it

class PicoScope(Instrument):
    """
    Interface for controlling PicoScope 5000 series oscilloscopes.
    
    This class provides methods to configure and control PicoScope oscilloscopes,
    including channel setup, triggering, and data acquisition.
    
    Parameters
    ----------
    resolution : str
        Resolution setting for the oscilloscope (e.g., '8', '12', '14', '15', '16') [bit]
    name : str, optional
        Custom name for the instrument instance
        
    Methods
    -------
    calculate_timebase(sampling_rate, bit_mode)
        Calculate the oscilloscope timebase for a given sampling rate
    get_sampling_rate(timebase)
        Calculate the sampling rate for a given timebase
    get_sampling_interval(timebase)
        Calculate the sampling interval for a given timebase
    get_best_timebase(channels)
        Get the best available timebase for the specified channels
    set_trigger(source, threshold, direction, delay, auto_trigger)
        Configure and enable the trigger
    disable_trigger()
        Disable the trigger
    acq_block(sample_rate, post_trigger_samples, pre_trigger_samples, memory_segment_index, time_out, downsampling_mode, downsampling_ratio)
        Acquire a block of data
    acq_streaming(sample_rate, sample_units, buffer_size)
        Start streaming data acquisition
    stop_streaming()
        Stop streaming data acquisition
    get_streamed_data()
        Get the acquired streaming data
    set_channel(channel, enabled, coupling, range, offset)
        Configure a channel
    send_TTL_trigger(voltage)
        Send a TTL trigger signal
    get_command_value(dict_name, key)
        Get a command value from the PicoStrings.json file
    disable_all_channels()
        Disable all channels
    initialize()
        Initialize the connection to the device
    reset()
        Reset the device
    close_connection()
        Close the connection to the device
    shutdown()
        Shut down the device
    kill()
        Stop the oscilloscope and close the connection
    info(verbose)
        Display status information
    """
    def __init__(self, resolution: str, name: str = None):
        # Set temp name if not provided
        name = name if name else "PicoScope_no_ID"
        super().__init__(name)
        self.pico_strings = None
        
        self.resolution = ps.PS5000A_DEVICE_RESOLUTION[self.get_command_value('DEVICE_RESOLUTION', resolution)]
        
        # Create chandle and status ready for use
        self.chandle = ctypes.c_int16()
        self.status = {}
        
        # Set up channel information storage
        # channel_info = {channel: {"enabled": bool, "coupling": str, "range": str, "offset": float}}
        self.channel_info = {}
        
        # Register the kill method to be called at exit
        atexit.register(self.kill)
                
    
    def calculate_timebase(self, sampling_rate: float, bit_mode = None) -> int:
        """
        Calculate the oscilloscope timebase value for a given sampling rate.
        
        Parameters
        ----------
        sampling_rate : float
            The desired sampling rate in Hz
        bit_mode : int, optional
            The resolution mode (8, 12, 14, 15, or 16).
            If None, uses the current resolution setting
            
        Returns
        -------
        int
            The calculated timebase value
            
        Raises
        ------
        ValueError
            If the sampling rate exceeds the maximum supported rate for the selected bit mode
        
        Notes
        -----
        The timebase calculation depends on the bit mode:
        
        - 8-bit: 
          - For sampling_interval < 8ns: timebase = log2(sampling_interval)
          - For sampling_interval >= 8ns: timebase = round(sampling_interval * 0.125) + 2
            
        - 12-bit:
          - For sampling_interval < 16ns: timebase = log2(sampling_interval * 0.5) + 1 (min value: 1)
          - For sampling_interval >= 16ns: timebase = round(sampling_interval * 0.0625) + 3
            
        - 14/15-bit:
          - For sampling_interval <= 8ns: timebase = 3
          - For sampling_interval > 8ns: timebase = round(sampling_interval * 0.125) + 2
            
        - 16-bit:
          - For sampling_interval <= 16ns: timebase = 4
          - For sampling_interval > 16ns: timebase = round(sampling_interval * 0.0625) + 3
        """
        bit_mode = self.get_command_value('RESOLUTION_BITMODE_INT', str(self.resolution)) if bit_mode is None else bit_mode
        
        # Convert sampling_rate to sampling_interval in nanoseconds
        sampling_interval = 1e9 / sampling_rate
 
        if bit_mode == 8:
            # 8-bit mode:
            if sampling_interval < 8:
                if sampling_interval < 1:
                    raise ValueError("Sampling rate must not exceed 1 GHz for 8-bit mode.")
                # Inverse: n = log2(sampling_interval)
                return int(round(np.log2(sampling_interval))) 
            else:
                # Inverse: n = sampling_interval * 1.25e-1 + 2
                return int(round(sampling_interval * 1.25e-1)) + 2

        elif bit_mode == 12:
            # 12-bit mode: available timebases are 1, 2, 3, then n>=4.
            if sampling_interval < 16:
                if sampling_interval < 2:
                    raise ValueError("Sampling rate must not exceed 500 MHz for 12-bit mode.")
                # Inverse: n = log2(sampling_interval * 5e-1) + 1
                dep = int(round(np.log2(sampling_interval * 5e-1))) + 1
                return dep if dep > 0 else 1
            else:
                # Inverse: n = sampling_interval * 0.0625 + 3
                return int(round(sampling_interval * 6.25e-2)) + 3

        elif bit_mode in (14, 15):
            if sampling_interval < 8:
                raise ValueError("Sampling rate must not exceed 125 MHz for 14/15-bit mode.")
            # 14-bit and 15-bit mode: valid timebases start at 3.
            if sampling_interval <= 8:
                return 3
            else:
                # Inverse: n = sampling_interval * 0.125 + 2
                return int(round(sampling_interval * 0.125)) + 2

        elif bit_mode == 16:
            if sampling_interval < 16:
                raise ValueError("Sampling rate must not exceed 62.5 MHz for 16-bit mode.")
            # 16-bit mode: valid timebases start at 4.
            if sampling_interval <= 16:
                return 4
            else:
                # Inverse: n = sampling_interval * 0.0625 + 3
                return int(round(sampling_interval * 6.25e-2)) + 3

        else:
            raise ValueError(f"Unsupported bit mode: {bit_mode}")
        
    def get_sampling_rate(self, timebase: int) -> float:
        """
        Calculate the sampling rate corresponding to a given timebase.
        
        Parameters
        ----------
        timebase : int
            The timebase value 
            
        Returns
        -------
        float
            The sampling rate in Hz
        """
        sampling_interval = self.get_sampling_interval(timebase)

        # Return the sampling rate in Hz (convert from ns to s first)
        return 1e9 / sampling_interval
    
    def get_sampling_interval(self, timebase: int) -> float:
        """
        Calculate the sampling interval corresponding to a given timebase.
        
        Parameters
        ----------
        timebase : int
            The timebase value
            
        Returns
        -------
        float
            The sampling interval in nanoseconds
            
        Raises
        ------
        ValueError
            If the sampling interval is less than 1 ns
        """
        sampling_interval = ctypes.c_float()
        max_samples = ctypes.c_int32()
        self.status["SamplingInterval"] = ps.ps5000aGetTimebase2(self.chandle, timebase, 1000, ctypes.byref(sampling_interval), ctypes.byref(max_samples), 0)
        
        if sampling_interval.value < 1:
            raise ValueError("Sampling interval should be at least 1 ns. Something went wrong. Try different timebase.")
        assert_pico_ok(self.status["SamplingInterval"])

        return sampling_interval.value
    
    def get_best_timebase(self, channels: list[str] =  ['A']) -> int:
        """
        Get the best available timebase for the specified channels.
        
        Parameters
        ----------
        channels : list of str, optional
            List of channel identifiers to consider (default is ['A'])
            
        Returns
        -------
        int
            The best available timebase value
        """
        status_key = "BestTimebase"
        timebase = ctypes.c_uint32()
        sampling_interval = ctypes.c_double()
        def concat_channels(channels):
            return sum([self.get_command_value('CHANNEL_FLAGS', ch) for ch in channels])
        self.status[status_key] = ps.ps5000aGetMinimumTimebaseStateless(self.chandle,
                                                        concat_channels(channels),
                                                        ctypes.byref(timebase),
                                                        ctypes.byref(sampling_interval),
                                                        self.resolution)
        assert_pico_ok(self.status[status_key])
        return timebase.value
    
    def set_trigger(self, source: str, threshold: float, direction: str, delay: int, auto_trigger: int = 1000):
        """
        Configure and enable the trigger.
        
        Parameters
        ----------
        source : str
            The trigger source channel ('A', 'B', 'C', or 'D')
        threshold : float
            The trigger threshold in millivolts
        direction : str
            The trigger direction ('ABOVE', 'BELOW', 'RISING', 'FALLING', or 'RISING_OR_FALLING')
        delay : int
            The trigger delay in samples
        auto_trigger : int, optional
            Number of milliseconds after which the scope will trigger automatically 
            if no trigger event occurs (default: 1000 ms)
            
        Raises
        ------
        ValueError
            If an unsupported trigger source or direction is provided
        """
        try:
            c_source = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{source}"]
        except KeyError:
            raise ValueError(f"Unsupported trigger source: {source}. Supported sources are A, B, C, D")
        direction = direction.upper()
        if direction not in ['ABOVE', 'BELOW', 'RISING', 'FALLING', 'RISING_OR_FALLING']:
            raise ValueError(f"Unsupported trigger direction: {direction}. Supported directions are ABOVE, BELOW, RISING, FALLING, RISING_OR_FALLING")
    
        direction = ps.PS5000A_THRESHOLD_DIRECTION[self.get_command_value('THRESHOLD_DIRECTION', direction)]
        
        # Get the channel range for the trigger source
        channel_range = self.channel_info[source]['range']
        
        # Convert threshold to ADC counts
        maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps.ps5000aMaximumValue(self.chandle, ctypes.byref(maxADC))
        assert_pico_ok(self.status["maximumValue"])
        threshold = int(mV2adc(threshold, channel_range, maxADC))
        
        self.status["trigger"] = ps.ps5000aSetSimpleTrigger(self.chandle, 1, c_source, threshold, direction, delay, auto_trigger)
        assert_pico_ok(self.status["trigger"])
    
    def disable_trigger(self):
        """
        Disable the trigger.
        """
        self.status["trigger"] = ps.ps5000aSetSimpleTrigger(self.chandle, 0, 0, 0, 0, 0, 0)
        assert_pico_ok(self.status["trigger"])
    
    def acq_block(
        self, 
        sample_rate: float, 
        post_trigger_samples: int, 
        pre_trigger_samples: int = 0, 
        memory_segment_index: int = 0, 
        time_out: int = 3000, 
        downsampling_mode: str = 'NONE', 
        downsampling_ratio: int = 1
    ):
        """
        Acquire a block of data from the oscilloscope.
        
        Parameters
        ----------
        sample_rate : float
            The desired sampling rate in Hz
        post_trigger_samples : int
            The number of samples to acquire after the trigger event
        pre_trigger_samples : int, optional
            The number of samples to acquire before the trigger event (default: 0)
        memory_segment_index : int, optional
            The index of the memory segment to use (default: 0)
        time_out : int, optional
            The timeout in milliseconds (default: 3000)
        downsampling_mode : str, optional
            The downsampling mode ('NONE', 'AVERAGE', 'MIN_MAX', 'DECIMATE', default: 'NONE')
        downsampling_ratio : int, optional
            The downsampling ratio (default: 1)
            
        Returns
        -------
        dict
            Dictionary containing the acquired data (in mV) for each channel and the time data
            
        Notes
        -----
        If no trigger is set, the number of samples acquired will be equal to 
        pre_trigger_samples + post_trigger_samples.
        
        Raises
        ------
        TimeoutError
            If the acquisition times out
        """
        no_of_samples = pre_trigger_samples + post_trigger_samples
        if no_of_samples == 0:
            raise Warning("No samples to acquire. Set pre_trigger_samples or post_trigger_samples. 100 samples will be acquired.")
            no_of_samples = 100
        downsampling_mode = downsampling_mode.upper()
        downsampling_mode = ps.PS5000A_RATIO_MODE[self.get_command_value('RATIO_MODE', downsampling_mode)]
        
        # Get timebase for the given sample rate
        timebase = self.calculate_timebase(sample_rate)
        
        # Hanldlers for returned values
        timeIndisposedMs = ctypes.c_int32()
        pParameter = ctypes.c_void_p()
        
        # Run the block acquisition
        self.status["runBlock"] = ps.ps5000aRunBlock(
            self.chandle,
            pre_trigger_samples,
            post_trigger_samples,
            timebase,
            ctypes.byref(timeIndisposedMs),
            memory_segment_index,
            None,
            ctypes.byref(pParameter)
        )
        assert_pico_ok(self.status["runBlock"])
        
        # Check for data collection to finish using ps5000aIsReady
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        start_time = time.time() # Set a timeout for the acquisition
        while ready.value == check.value:
            self.status["isReady"] = ps.ps5000aIsReady(self.chandle, ctypes.byref(ready))
            if time.time() - start_time > time_out / 1000:  # Convert milliseconds to seconds
                raise TimeoutError("Block acquisition timed out.")
               
        assert_pico_ok(self.status["isReady"])
        
        # Calculate the number of samples to be collected
        if downsampling_mode != 0:
            no_of_samples = no_of_samples // downsampling_ratio if downsampling_ratio > 1 else no_of_samples
        
        # Create max and min buffers ready for assigning pointers for data collection for active channels
        buffers = {}
        for ch in self.channel_info.keys():
            if self.channel_info[ch]['enabled']:
                buffers[ch] = {'Max': (ctypes.c_int16 * no_of_samples)(), 'Min': (ctypes.c_int16 * no_of_samples)()}
            
        # Set data buffer location for data collection from active channels
        for ch in buffers.keys():
            source = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{ch}"]
            status_key = f"setDataBuffers{ch}"
            self.status[status_key] = ps.ps5000aSetDataBuffers(
                self.chandle,
                source,
                ctypes.byref(buffers[ch]['Max']),
                ctypes.byref(buffers[ch]['Min']),
                no_of_samples,
                memory_segment_index,
                downsampling_mode
            )
            assert_pico_ok(self.status[status_key])
        
        overflow = ctypes.c_int16()
        cmaxSamples = ctypes.c_int32(no_of_samples)
        
        self.status["getValues"] = ps.ps5000aGetValues(self.chandle, 0, ctypes.byref(cmaxSamples), downsampling_ratio, downsampling_mode, memory_segment_index, ctypes.byref(overflow))
        assert_pico_ok(self.status["getValues"])
        if overflow.value != 0:
            print(f"Overflow occurred: {overflow.value} samples lost. type: {type(overflow)}, {type(overflow.value)}")
            self.status["overflow"] = overflow.value
        
        # convert ADC counts data to mV
        maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps.ps5000aMaximumValue(self.chandle, ctypes.byref(maxADC))
        assert_pico_ok(self.status["maximumValue"])
        data = {}
        for ch in buffers.keys():
            data[ch] =  adc2mV(buffers[ch]['Max'], self.channel_info[ch]['range'], maxADC)

        # Create time data
        data['time'] = np.linspace(0, (cmaxSamples.value - 1) / sample_rate, cmaxSamples.value)
        
        return data
    
    
    def acq_streaming(self, sample_rate: float = 4e6,  # 4 MHz default, corresponds to 250ns interval
                    sample_units = ps.PS5000A_TIME_UNITS['PS5000A_NS'], 
                    buffer_size: int = 500):
        """
        Start streaming data acquisition continuously.
        
        Sets up a buffer for channel A and registers a callback that appends each received 
        chunk to self._streamed_data. The acquisition continues until stop_streaming() is called.
        
        Parameters
        ----------
        sample_rate : float, optional
            The desired sampling rate in Hz (default: 4e6, or 4 MHz)
        sample_units : int or str, optional
            The time unit for the sample interval
        buffer_size : int, optional
            Size of the buffer for data acquisition (default: 500)
        """
        sample_units = sample_units if isinstance(sample_units, int) else ps.PS5000A_TIME_UNITS[sample_units]
        self._streaming_stop = False
        self._streamed_data = []  # reset storage
        self._buffer_size = buffer_size
        
        # Convert sample_rate to sample_interval in ns
        sample_interval = int(1e9 / sample_rate)
        
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
        print(f"Streaming started at sample interval {sample_interval_ct.value} ns (rate: {1e9/sample_interval_ct.value:.2f} Hz).")

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
        Stop streaming data acquisition.
        
        Signals to stop the acquisition and waits for the streaming thread to finish.
        """
        self._streaming_stop = True
        if self._streaming_thread is not None:
            self._streaming_thread.join()

    def get_streamed_data(self):
        """
        Get the acquired streaming data.
        
        Returns
        -------
        numpy.ndarray
            Concatenated array of all acquired data chunks, or empty array if no data
        """
        if self._streamed_data:
            return np.concatenate(self._streamed_data)
        else:
            return np.array([], dtype=np.int16)
        
    def set_channel(self, channel, enabled, coupling, range, offset):
        """
        Configure a channel on the oscilloscope.
        
        Parameters
        ----------
        channel : str
            Channel identifier ('A', 'B', 'C', 'D')
        enabled : bool
            Whether to enable the channel
        coupling : str
            Channel coupling type ('DC' or 'AC')
        range : str
            Channel voltage range in volts (e.g., '2V') or millivolts (e.g., '20MV')
        offset : float
            Channel offset in volts
            
        Raises
        ------
        ValueError
            If an invalid range is provided
        """
        enabled = 1 if enabled else 0
        status_key = f"setCh{channel}"
        try:
            range_value = ps.PS5000A_RANGE[self.get_command_value('RANGE', range)]
        except KeyError:
            raise ValueError(f"Invalid range: {range}. Check the available ranges on the json file.")
        self.status[status_key] = ps.ps5000aSetChannel(self.chandle,
                                                    ps.PS5000A_CHANNEL[f'PS5000A_CHANNEL_{channel}'],
                                                    enabled,
                                                    ps.PS5000A_COUPLING['PS5000A_DC' if coupling == 'DC' else 'PS5000A_AC'],
                                                    range_value,
                                                    offset)
        assert_pico_ok(self.status[status_key])
        
        # Store the channel information
        self.channel_info[channel] = {'enabled': enabled, 'coupling': coupling, 'range': range_value, 'offset': offset}

    def send_TTL_trigger(self):
        """
        Send a TTL trigger signal.
        
        Parameters
        ----------
        voltage : float, optional
            The voltage level for the trigger signal in volts (default: 2V)
        """
        # Set the trigger output to the specified voltage
        set_sig_gen_voltage = lambda : ps.ps5000aSetSigGenBuiltInV2(self.chandle, 
                                                                     ctypes.c_int32(int(0e6)), # offsetVoltage
                                                                     ctypes.c_uint32(int(1e6)),      # pkToPk
                                                                     ctypes.c_int32(1),       # waveType (SQUARE)
                                                                     ctypes.c_double(8),       # startFrequency
                                                                     ctypes.c_double(8),       # stopFrequency
                                                                     ctypes.c_double(0),       # increment
                                                                     ctypes.c_double(1),     # dwellTime
                                                                     ctypes.c_int32(0),       # sweepType (UP)
                                                                     ctypes.c_int32(0),       # operation (ADD)
                                                                     ctypes.c_uint32(1),      # shots
                                                                     ctypes.c_uint32(0),      # sweeps
                                                                     ctypes.c_int32(0),       # triggerType (RISING)
                                                                     ctypes.c_int32(4),       # triggerSource (SOFTWARE TRIGGER)
                                                                     ctypes.c_int16(0))       # extInThreshold
        
        self.status["setSigGenVoltage"] = set_sig_gen_voltage()  # Convert to uV
        assert_pico_ok(self.status["setSigGenVoltage"])
        # Use state '2' (PS5000A_SIGGEN_TRIG_RISING) to apply the software trigger
        self.status["launchTrigger"] = ps.ps5000aSigGenSoftwareControl(self.chandle, ctypes.c_int16(2))
        assert_pico_ok(self.status["launchTrigger"])

        
        
        
    # UTILITY METHODS ---------------------------------------------------------
    def get_command_value(self, dict_name, key):
        """
        Get a command value from the PicoStrings.json file.
        
        Parameters
        ----------
        dict_name : str
            The name of the dictionary in the PicoStrings.json file
        key : str
            The key to look up in the dictionary
            
        Returns
        -------
        object
            The value associated with the key
        """
        if self.pico_strings is None:  
            # Read the json PicoStrings file
            python_file_dir = os.path.dirname(__file__)
            with open(os.path.join(python_file_dir, 'cfgs/PicoStrings.json')) as f:
                self.pico_strings = json.load(f)
                
        # Get the value
        return self.pico_strings[dict_name][key]
    
    def disable_all_channels(self):
        """
        Disable all input channels (A, B, C, D).
        """
        for channel in ['A', 'B', 'C', 'D']:
            self.set_channel(channel, False, 'DC', '20MV', 0)
            
            
    # IMPLEMENTATION OF THE ABSTRACT METHODS ----------------------------------

    def initialize(self):
        """
        Initialize the connection to the device.
        
        Opens the PicoScope device with the specified resolution.
        After connection, disables all channels.
        
        Raises
        ------
        Exception
            If the device cannot be opened
        """
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
        # Disable all channels
        self.disable_all_channels()
    
    def _activate(self): # Dummy method
        """
        Placeholder method for the abstract _activate method.
        """
        pass
    
    def reset(self):
        """
        Reset the device to its initial state.
        
        Disables the trigger, closes the connection, clears the status,
        and reinitializes the device.
        """
        self.disable_trigger()
        self.close_connection()
        self.status = {}
        self.channel_info = {}
        self.pico_strings = None
        self.initialize()
    
    def close_connection(self):
        """
        Close the connection to the device.
        """
        # Disconnect the scope
        self.status["close"] = ps.ps5000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])
    
    def shutdown(self):
        """
        Shut down the device cleanly.
        
        Calls kill() and notifies the user to unplug the PicoScope.
        """
        self.kill()
        print("Unplug PicoScope")
        
    def kill(self):
        """
        Stop the oscilloscope and close the connection.
        """
        # Stop the scope
        self.status["stop"] = ps.ps5000aStop(self.chandle)
        #assert_pico_ok(self.status["stop"])
    
        self.close_connection()
    
    def info(self, verbose=False):
        """
        Display status information about the device.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, provide more detailed information (default: False)
        """
        # Display status returns
        print(self.status)
    
    def __del__(self):
        """
        Destructor to ensure the connection is closed when the object is deleted.
        """
        # Ensure the connection is closed when the object is deleted
        if self.status['close'] == None and self.status['stop'] == None :
            self.close_connection()
