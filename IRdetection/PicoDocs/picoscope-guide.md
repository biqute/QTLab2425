# PicoScope 5444D MSO - Programming Guide

Purpose: this guide helps researchers and engineers get started with the PicoScope 5444D MSO. It describes the instrument, shows how to install the software/SDK, explains how to program the scope via the official PicoSDK (old C-based wrapper) and the newer pyPicoSDK package, and shows how to use the custom library in the IRdetection repository.

## Introduction

The PicoScope 5444D MSO belongs to Pico Technology's 5000 D-series of FlexRes oscilloscopes. These devices combine high bandwidth with flexible resolution. Key hardware characteristics include:

- Analog channels: four independent channels with $\mathbf{2 0 0 ~ M H z}$ analog bandwidth and selectable input ranges from $\pm 10 \mathrm{mV}$ to $\pm 20 \mathrm{~V}$. The scope's FlexRes architecture allows hardware resolution to be switched between 8-16 bits; lower resolutions give higher maximum sampling rates while 16-bit mode provides high precision.
- Mixed-signal capability: the MSO version adds $\mathbf{1 6}$ digital inputs. Each input can acquire signals up to 100 MHz and minimum pulse widths down to 5 ns. Two independent threshold settings allow TTL/CMOS and other logic levels.
- Sample rate and memory: in 8-bit mode the scope can sample at $\mathbf{1}$ GS/s with a single channel. With two analog channels enabled the maximum is $500 \mathrm{MS} / \mathrm{s}$ and decreases as more channels are used. In 16-bit mode the maximum sampling rate is $62.5 \mathrm{MS} / \mathrm{s}$. The scope's deep memory (128512 MS) supports long captures and rapid block mode acquisitions.
- Other features: built-in arbitrary waveform generator (AWG), advanced digital triggering, serial protocol decoding, USB 3.0 for high-speed streaming and PC-powered operation.

The 5444D MSO uses the ps5000a driver. Pico Technology's knowledge-base lists the current 5000 series models (5242D, 5243D, 5244D, 5442D, 5443D and 5444D) and notes that they all use the ps5000a API. The older 5000 " A " and " B " models use different drivers (ps4000a/ps5000), so ensure you download the correct SDK.

Image placeholder:

PicoScope 5444D MSO - placeholder

Replace <IMAGE_PLACEHOLDER> with a picture of your instrument.

Summary: The 5444D MSO offers four 200 MHz analog channels, 16 digital inputs, FlexRes resolution (816 bit), deep memory and USB 3.0 streaming. It uses the ps5000a API and requires the PicoSDK drivers to operate.

## Quickstart

## 1 - Install PicoScope software and SDK

1. Download and install PicoScope 7 and PicoSDK. The Pico knowledge-base notes that pyPicoSDK is a wrapper around PicoSDK, so the SDK must be installed even if you use Python. If PicoScope 7 is already installed, it includes the latest drivers; otherwise:
2. Visit picotech.com/downloads, select your scope series (PicoScope 5000 series) and download the PicoSDK installer for your operating system.
3. Run the installer (Windows/macOS) and accept the default options. Linux users should visit the Linux downloads page and install the driver package via apt or similar.
4. Verify installation. After installation you can run PicoScope 7. Use the More $\rightarrow$ About PicoScope 7 menu to check the version and update if required.
5. Connect the scope. Connect the USB 3.0 cable and (if using the AWG) the mains power. When first connected, the OS will load drivers.

## 2 - Install pyPicoSDK

pyPicoSDK is Pico Technology's modern Python wrapper over the C-based SDK. It simplifies many tasks and provides type-safe enumerations. However, at the time of writing (26 September 2025) it only supports the PicoScope 6000E and 3000E series; support for the 5000D series is not yet available. You may still install the package to experiment, but you cannot control a 5444D with it until a ps5000a driver is added.

Installation:

```
python -m pip install pypicosdk
```

After installation you can verify the version with python -c "import pypicosdk; print(pypicosdk.version.VERSION)".

## 3 - Create a pyPicoSDK script (for supported models)

Although the 5444D is not yet supported, it is useful to see how pyPicoSDK scripts are structured. The simple_block_capture.py example in Pico's GitHub repository shows typical usage:

```
import pypicosdk as psdk
from matplotlib import pyplot as plt
# Number of samples and sample rate (in MS/s)
samples = 5000
sample_rate_msps = 500
# Create a scope object - replace ps6000a() with the driver matching your device
scope = psdk.ps6000a() # for 6000E series; psospa() for 3000E
```

```
scope.open_unit() # connect to the scope
# Set the built-in signal generator
scope.set_siggen(frequency=1_000_000, pk2pk=0.8,
    wave_type=psdk.WAVEFORM.SINE) \227158665319482†L683-L688\
# Turn on channel A and set a 1 V range
scope.set_channel(channel=psdk.CHANNEL.A, range=psdk.RANGE.V1)
\227158665319482田L697-L703|
# Configure a rising-edge trigger on channel A at 0 mV
scope.set_simple_trigger(channel=psdk.CHANNEL.A, threshold_mv=0)
\227158665319482男L697-L705\
# Convert desired sample rate to a timebase index
# (msps = megasamples per second)
timebase = scope.sample_rate_to_timebase(sample_rate=500,
    unit=psdk.SAMPLE_RATE.MSPS)
\227158665319482田L714-L725\
# Capture data (block mode)
channel_buffer, time_axis = scope.run_simple_block_capture(timebase, samples)
\227158665319482田L741-L748\\
# Plot the captured data
plt.plot(time_axis, channel_buffer)
plt.xlabel('Time (s)')
plt.ylabel('ADC Counts')
plt.show()
```

For a 5444D，replace ps6000a（）with the appropriate class once the ps5000a driver is added（e．g． scope $=$ psdk．ps5000a（））and ensure the sample rate does not exceed the scope＇s specification．

Summary：The PicoSDK and PicoScope 7 software must be installed before using any Python wrapper． pyPicoSDK currently supports 6000E／3000E scopes．The example shows typical steps：create a scope object，open the unit，configure channels and triggers，set the timebase and acquire data．

## Python Wrapper（Old API）

When pyPicoSDK does not support a device（such as the 5444D MSO），you can program the scope using the C－based PicoSDK through the picosdk Python wrappers．These wrappers mirror the C API closely， so you need to map constants and structures correctly．

## Installing the picosdk wrapper

The picosdk Python wrappers are available on GitHub (picosdk-python-wrappers). Clone the repository and ensure that the PicoSDK drivers are installed. You can then run:

```
git clone https://github.com/picotech/picosdk-python-wrappers.git
cd picosdk-python-wrappers
python -m pip install -r requirements.txt
```

Alternatively, copy the picosdk directory into your project.

## Basic usage

The wrappers expose driver modules, e.g. picosdk.ps5000a for the 5000 D series. The typical workflow is:

1. Open the device: call ps5000a0penUnit (via ps.open_unit) to obtain a handle . You can query the variant information using ps5000aGetUnitInfo with the PICO_VARIANT_INFO parameter; the API will return the model number (e.g. 5444D).
2. Set up channels and resolution: call ps5000aSetDeviceResolution to choose the resolution (8-16 bits). Use ps5000aSetChannel to enable an analog channel, set coupling (AC/DC), input range and analogue offset.
3. Configure digital ports (MSO models): call ps5000aSetDigitalPort to enable digital inputs and set logic thresholds and directions.
4. Configure triggers: use ps5000aSetSimpleTrigger or ps5000aSetTriggerChannelConditions and related functions to define analog/digital trigger conditions. Trigger directions (rising, falling, above, below, etc.) are enumerated values; in our custom library these are stored in PicoStrings.json (see next section).
5. Determine the timebase: call ps5000aGetTimebase to convert a desired sample interval to a timebase index. Use the formulas described in the API manual or our custom library to ensure that the requested rate is within the scope's limits.
6. Run a capture: choose block mode ( ps5000aRunBlock ) for finite captures or streaming mode ( ps5000aRunStreaming ) for continuous acquisition. In block mode you must provide pre- and post-trigger sample counts and a timebase index; after ps5000aRunBlock completes you can call ps5000aGetValues to retrieve the samples.
7. Retrieve data and downsample: ps5000aGetValues accepts a downsample ratio and PS5000A_RATIO_MODE to perform decimate, average or aggregate downsampling. The enumerated values PS5000A_RATIO_MODE_NONE, AGGREGATE, DECIMATE, AVERAGE and DISTRIBUTION are defined in the API. Use these to reduce large datasets or to obtain min/max pairs for plotting.
8. Close the device: call ps5000aCloseUnit when finished.

## Mapping constants

Because the C API uses numeric constants, the Python wrapper provides enumerations. The PicoStrings.json file in the IRdetection repository maps human-readable strings to the ps5000a
constants. For example, the dictionary DEVICE_RESOLUTION maps bit-depth strings (" 8 ", " 12 ", " 14 ", " 15 ", "16") to API constants like PS5000A_DR_8BIT. Similarly, RANGE maps voltage ranges ("10MV", "20MV", ..., "50V") to PS5000A_10MV, PS5000A_20MV, etc. Trigger directions and digital channel flags are also enumerated. Use these dictionaries to translate intuitive parameters into the C API values when calling wrapper functions.

## Example: block mode acquisition

Below is a minimal example that uses the picosdk wrapper to capture a waveform on channel A with the 5444D MSO. Error handling is omitted for clarity.

```
import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, adc2mV
# Open the device
device_handle = ctypes.c_int16()
status = ps.ps5000aOpenUnit(ctypes.byref(device_handle), None)
assert_pico_ok(status)
# Choose 12-bit resolution
resolution = ps.PS5000A_DEVICE_RESOLUTION['PS5000A_DR_12BIT']
status = ps.ps5000aSetDeviceResolution(device_handle, resolution)
assert_pico_ok(status)
# Enable channel A, DC coupling, t2 V range
status = ps.ps5000aSetChannel(device_handle,
ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'], True,
    ps.PS5000A_COUPLING['PS5000A_DC'],
ps.PS5000A_RANGE['PS5000A_2V'], 0.0)
assert_pico_ok(status)
# Set a rising-edge trigger at 0 V on channel A
status = ps.ps5000aSetSimpleTrigger(device_handle, 1, # enable
        ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
        0, # threshold in ADC counts
ps.PS5000A_THRESHOLD_DIRECTION['PS5000A_RISING'],
        0, # delay
        1000) # auto-trigger timeout (ms)
assert_pico_ok(status)
# Calculate a timebase for 125 MS/s (8 ns interval)
timebase = 3 # from formula or ps5000aGetTimebase; for 12-bit mode this yields
125 MS/s
```

```
# Set number of pre-/post-trigger samples
pre_trigger = 0
post_trigger = 5000
# Run the block capture
status = ps.ps5000aRunBlock(device_handle, pre_trigger, post_trigger, timebase,
None, 0, None, None)
assert_pico_ok(status)
# Wait until ready and retrieve data
ready = ctypes.c_int16(0)
while ready.value == 0:
    ps.ps5000aIsReady(device_handle, ctypes.byref(ready))
# Prepare buffers and call ps5000aGetValues
buffer_a = (ctypes.c_int16 * post_trigger)()
ps.ps5000aSetDataBuffer(device_handle, ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
buffer_a, post_trigger, 0, 0)
samples = ctypes.c_uint32(post_trigger)
status = ps.ps5000aGetValues(device_handle, 0, ctypes.byref(samples), 1,
        ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'],
0, None)
assert_pico_ok(status)
# Convert ADC counts to millivolts
voltages = adc2mV(buffer_a, ps.PS5000A_RANGE['PS5000A_2V'], resolution)
# Close the device
ps.ps5000aCloseUnit(device_handle)
```

This example uses dictionary lookups from the ps5000a module for channel names, coupling, ranges and threshold directions. The adc 2 mV function converts raw ADC counts to physical units.

Summary: The old picosdk wrapper lets you program the 5444D directly. Follow the C API closely: open the unit, configure channels and triggers, calculate the timebase, run the acquisition (block or streaming), retrieve data with ps5000aGetValues and close the device. Use enumeration dictionaries or the PicoStrings.json mapping to translate intuitive parameters into API constants.

## Our Python Library (IRdetection)

The IRdetection repository contains a custom PicoScope class designed to simplify control of PicoScope 5000 series oscilloscopes. The library wraps the picosdk functions and provides high-level methods for channel setup, triggering, sampling and streaming.

## Features and architecture

The PicoScope class (see irdetection/instruments/PicoScope.py) inherits from an Instrument base class and is instantiated with a chosen resolution (e.g., ' 8 ' , ' 12 ' , ' 14 ' , ' 15 ' or '16' ). Its docstring states that it provides an interface for controlling PicoScope 5000 series oscilloscopes, including channel setup, triggering and data acquisition. Key methods include:

- calculate_timebase(sampling_rate, bit_mode) - calculates the timebase integer for a desired sampling rate. The method contains formulas for each resolution mode. For example, in 8-bit mode a sampling interval < 8 ns uses ( $\log 2$ (sampling_interval), while larger intervals use (sampling_interval $\times 0.125$ ) ). Similar formulas are implemented for 12-bit, 14/15-bit and 16-bit modes.
- get_sampling_rate(timebase) and get_sampling_interval(timebase) - convert between timebase indices and sampling intervals.
- set_channel(channel, enabled, coupling, range, offset) - enable or disable an analog channel, set AC/DC coupling, input range and offset.
- set_trigger(source, threshold, direction, delay, auto_trigger) - configure a trigger on a specific channel using threshold and direction enums.
- acq_block() - perform a block capture. It wraps ps5000aRunBlock and ps5000aGetValues, automatically allocating buffers and converting results to numpy arrays.
- acq_streaming() / get_streamed_data() - start streaming mode and retrieve data continuously.
- send_TTL_trigger() - generate a TTL trigger output.
- disable_all_channels(), initialize(), reset(), close_connection() - convenience functions for housekeeping.


## Simplified constants via PicoStrings.json

The library uses a JSON configuration file (irdetection/instruments/cfgs/PicoStrings.json ) to map human-readable strings to ps5000a constants. For example, DEVICE_RESOLUTION maps '8' $\rightarrow$ PS5000A_DR_8BIT, ' 12 ' $\rightarrow$ PS5000A_DR_12BIT, etc. Other dictionaries define coupling (AC/DC), ranges ( 10 mV to 50 V ), trigger directions (rising, falling, above, below, etc.) and digital channel enumerations. When you call set_channel or set_trigger, the library looks up the corresponding constant automatically.

## Usage example

The following example demonstrates how to use the custom library to capture a single waveform on a 5444D MSO. This code assumes the irdetection package is installed in your Python environment.

```
from irdetection.instruments.PicoScope import PicoScope
# Create a PicoScope object with 12-bit resolution
ps = PicoScope(resolution='12')
# Initialize the connection
```

```
ps.initialize()
# Enable channel A with DC coupling, t2 V range and zero offset
ps.set_channel(channel='A', enabled=True, coupling='DC', range='2V', offset=0.0)
# Disable unused channels (recommended to optimise sample rate)
ps.disable_all_channels()
# Configure a simple rising-edge trigger on channel A at 0 V with a 1 ms
auto-trigger timeout
ps.set_trigger(source='A', threshold=0.0, direction='RISING', delay=0,
auto_trigger=1_000)
# Acquire 5 000 post-trigger samples at 125 MS/s using block mode
sample_rate = 125e6 # 125 MS/s
pre_trigger = 0
post_trigger = 5_000
# Calculate timebase using the library's formula
timebase = ps.calculate_timebase(sample_rate, bit_mode=None)
# Run a block capture
timeout_ms = 5_000
downsampling_mode = 'NONE'
downsampling_ratio = 1
waveform = ps.acq_block(sample_rate=sample_rate,
    post_trigger_samples=post_trigger,
    pre_trigger_samples=pre_trigger,
    memory_segment_index=0,
    time_out=timeout_ms,
    downsampling_mode=downsampling_mode,
    downsampling_ratio=downsampling_ratio)
# waveform is a dictionary with channel names as keys and numpy arrays as values
channel_a_data = waveform['A']
# Clean up
ps.close_connection()
```

How it works: The PicoScope class hides the complexity of the C API. When you call initialize(), the library opens the unit and sets the device resolution. set_channel() translates the string parameters via PicoStrings.json and calls ps5000aSetChannel. The calculate_timebase() method uses built-in formulas to compute a valid timebase index for the requested sample rate. acq_block() triggers a capture, waits for completion, retrieves the data and returns it as numpy arrays. You can then plot or process the data in Python.

Summary: The custom PicoScope library in IRdetection abstracts the ps5000a API. It uses a JSON file to map human-readable strings to driver constants and implements functions for timebase calculation, channel configuration, triggering and data acquisition. This greatly reduces boilerplate code when working with a 5444D MSO.

## Final Remarks

- Always consult the official PicoScope 5000 Series (A API) Programmer's Guide for detailed information about each function. This guide describes data acquisition modes, buffer management, trigger setups and enumeration values; for example, the ps5000aGetValues function retrieves block-mode data and supports multiple downsampling modes.
- The 5444D MSO is a mixed-signal oscilloscope; digital channels are optional. You must call ps5000aSetDigitalPort and set thresholds to use them.
- For highest sample rates, disable unused channels and choose lower resolutions (8 or 12 bits).
- At the time of writing, pyPicoSDK does not yet support the ps5000a driver. Use the C-based wrapper or the custom library until Pico Technology adds support.

By following this guide, you can set up the PicoScope 5444D MSO, install the necessary software and control it via Python. Whether you choose the low-level wrappers or the high-level custom library, the 5000 D-series offers powerful mixed-signal measurement capabilities for research and development.

PicoScope 5000D Series oscilloscope data sheet
https://www.picotech.com/download/datasheets/picoscope-5000d-series-data-sheet.pdf
PicoSDK and PicoScope models | Pico Knowledge Base
https://www.picotech.com/library/knowledge-bases/oscilloscopes/picosdk-and-picoscope-models
pyPicoSDK: Getting started with Python and PicoSDK
https://www.picotech.com/library/knowledge-bases/oscilloscopes/pypicosdk-get-started
raw.githubusercontent.com
https://raw.githubusercontent.com/biqute/QTLab2425/main/IRdetection/irdetection/instruments/cfgs/PicoStrings.json
raw.githubusercontent.com
https://raw.githubusercontent.com/biqute/QTLab2425/main/IRdetection/irdetection/instruments/PicoScope.py

