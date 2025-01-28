#import niscope
import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import matplotlib.pyplot as plt

with nidaqmx.Task() as task:

    task.ai_channels.add_ai_voltage_chan("PXI1Slot2/ai0")

    task.timing.cfg_samp_clk_timing(int(2.5e6), sample_mode=AcquisitionType.FINITE, samps_per_chan=int(1e6))

    data = task.read(READ_ALL_AVAILABLE)

    plt.plot(data)

    plt.ylabel('Amplitude')

    plt.title('Waveform')
    
    # task.ai_channels.add_ai_voltage_chan("PXI1Slot2/ai1")
    
    # data2 = task.read(READ_ALL_AVAILABLE)
    
    # plt.plot(data2, 'r')

    plt.show()
# with nidaqmx.Task() as task:

#     task.ai_channels.add_ai_voltage_chan("PXI1Slot2/ai0", min_val=-10.0, max_val=10.0)

#     result = task.read()
#     print(result)
#session = niscope.Session("pippo", reset_device=True, id_query=True)