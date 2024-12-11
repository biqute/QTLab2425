
# Comandi utili

## Activate python environment
```shell
qubit3d\Scripts\activate
```

## Logout git
```shell
git config --global user.email ""  
git config --global user.name ""  
echo NAME:
git config user.name
echo EMAIL:
git config user.email
```

## Login git
```shell
git config --global user.email "maregariccardo.rm0@gmail.com"  
git config --global user.name "RiCkymare00"
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

## Link utili
- <https://peps.python.org/pep-0008/>
- <https://www.keysight.com/us/en/assets/7018-03314/data-sheets/5990-9783.pdf>
- <https://www.keysight.com/us/en/assets/9921-02561/programming-guides/FFProgrammingHelp.pdf>
- <https://micro-electronics.ru/upload/iblock/db7/db7419e1ab4b21017b4a7493870abd8b.pdf>
- <https://download.tek.com/manual/MDO4000-MSO4000B-and-DPO4000B-Oscilloscope-Programmer-Manual.pdf>

## TODO
- `VNA.py`:
    - plotting functions??
    - test if sleep() makes pyvisa work for LO
- `LO.py`:
    - turn_on() and turn_off() seem to do nothing
    - check the existence of the COM, if it doesn't return list of available COMs
- a -> polynomial in f (up to third order)