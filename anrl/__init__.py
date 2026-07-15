"""shadow-moment-crossover (``anrl``).

Code for the exact variance law and the crossover to collective measurement in
shadow-based estimation of quantum state moments. Subpackages:

- ``anrl.theory``    : the exact Hoeffding/U-statistic variance law, the two
  collective bias laws, the crossover, and the threshold ``M*``
- ``anrl.benchmark`` : Monte-Carlo estimators (single-copy shadows, collective
  SWAP test), noise channels, and moment operators
- ``anrl.physics``   : state ensembles and Pauli machinery shared by the above
- ``anrl.hardware``  : Open Quantum / Rigetti Cepheus backend, circuit builders,
  and the destructive-SWAP protocol
- ``anrl.figures``   : publication figure builders
"""

__version__ = "0.0.0"
