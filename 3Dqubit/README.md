
# Useful commands

## Python environment
Activate
```shell
qubit3d\Scripts\activate
```
Deactivate
```shell
qubit3d\Scripts\deactivate.bat
```

## Logout git
```shell
git config user.email "" --replace-all
git config user.name "" --replace-all
echo NAME:
git config user.name
echo EMAIL:
git config user.email
```

## Login git
```shell
git config user.email "zanindavide02@gmail.com" --replace-all
git config user.name "ZaninDavide" --replace-all
echo NAME:
git config user.name
echo EMAIL:
git config user.email
```


## Commit git
```shell
git pull
git commit -a -m "..."
git push
```

# Useful links
- <https://peps.python.org/pep-0008/>
- <https://www.keysight.com/us/en/assets/7018-03314/data-sheets/5990-9783.pdf>
- <https://www.keysight.com/us/en/assets/9921-02561/programming-guides/FFProgrammingHelp.pdf>
- <https://micro-electronics.ru/upload/iblock/db7/db7419e1ab4b21017b4a7493870abd8b.pdf>
- <https://download.tek.com/manual/MDO4000-MSO4000B-and-DPO4000B-Oscilloscope-Programmer-Manual.pdf>
- <https://github.com/Wheeler1711/submm_python_routines/blob/main/submm/demo/res_fit.ipynb>
- <https://markimicrowave.com/>
- [An analysis method for asymmetric resonator transmission applied to superconducting devices](https://arxiv.org/pdf/1108.3117)
- [Materials loss measurements using superconducting microwave resonators](https://arxiv.org/pdf/2006.04718)
- [AWG device manual](https://www.silcon.cz/download/SDG6000X_UserManual.pdf)
- [AWG programming manual](https://tm-co.co.jp/wp/wp-content/uploads/2022/10/SDG_Programming-Guide_PG02-E05C.pdf)
- [AWG implementation](https://github.com/sgoadhouse/awg_scpi/tree/main)
- [HEMT LNF-LNC4_16C Datasheet](https://lownoisefactory.com/wp-content/uploads/2023/03/lnf-lnc4_16c.pdf)
- [EM notes](https://web.archive.org/web/20240225035303/https://www.ece.rutgers.edu/~orfanidi/ewa/ch01.pdf)
- <https://biqute.github.io/qtics/instruments/triton.html>
- [Measurement of Resonant Frequency and Quality Factor of Microwave Resonators: Comparison of Methods](https://www.researchgate.net/publication/1947194_Measurement_of_Resonant_Frequency_and_Quality_Factor_of_Microwave_Resonators_Comparison_of_Methods)
- <https://rsl.yale.edu/sites/default/files/2024-08/2022-Thesis-Lev-Krayzman.pdf>
- <https://rsl.yale.edu/sites/default/files/2024-08/2017-RSL-Thesis-Teresa-Brecht-Final_ScreenVersion.pdf>
- <https://rsl.yale.edu/sites/default/files/2024-08/2016-RSL-Thesis-Matt-Reagor.pdf>
- <https://rsl.yale.edu/sites/default/files/2024-08/2023-Thesis-Suhas%20Ganjam.pdf>

# TODO
- `VNA.py`:
    - plotting functions??
    - test if sleep() makes pyvisa work for LO
- `LO.py`:
    - turn_on() and turn_off() seem to do nothing
    - check the existence of the COM, if it doesn't return list of available COMs
- `EthernetDevice.py`:
    - print executed command before getting OPC response
- `Fitter.py`:
    - allow use of already calculated derived params for following params (maybe)
    - check that initial values of params are inside the given range
- `cavity_iq.py`:
    - estimate $Q_l$ instead of $Q_i$ with $\omega_0 / \Delta \omega$
- `rettaroli_fit.py`:
    - after fitting the resonance, plot magnitude + phase (Bode plot)
- `estimate_rettaroli_params.py`:
    - allow for different sets of frequency points for $S_{11}$ and $S_{21}$ data 
