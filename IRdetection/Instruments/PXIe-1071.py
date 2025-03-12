import niscope
import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import matplotlib.pyplot as plt

with nidaqmx.Task() as task:

    task.ai_channels.add_ai_voltage_chan("PXI1Slot2/ai0")

    task.timing.cfg_samp_clk_timing(100000.0, sample_mode=AcquisitionType.FINITE, samps_per_chan=1000)

    data = task.read(READ_ALL_AVAILABLE)

    plt.plot(data)

    plt.ylabel('Amplitude')

    plt.title('Waveform')

    plt.show()
# with nidaqmx.Task() as task:

#     task.ai_channels.add_ai_voltage_chan("PXI1Slot2/ai0", min_val=-10.0, max_val=10.0)

#     result = task.read()
#     print(result)
#session = niscope.Session("pippo", reset_device=True, id_query=True)