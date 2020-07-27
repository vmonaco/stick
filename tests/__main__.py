import os
import tempfile

from test.stick import *
from test.algorithms import *
from test.memory import *
from test.primitives import *
from test.relational import *
from test.routing import *
from test.misc import *

FIGURES_DIR = os.path.join(tempfile.gettempdir(), 'stick-figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Run all tests to generate figures and results
if __name__ == '__main__':
    print('Saving figures to:', FIGURES_DIR)

    # STICK network composition and value encoding/decoding
    test_ge_synapse(out=FIGURES_DIR)
    test_timesteps()
    test_nested()
    test_encoding()

    # Relational
    test_minimum()
    test_maximum()

    # Primitives
    test_iterator(out=os.path.join(FIGURES_DIR, 'iterator-chronogram.pdf'))
    test_addresser(out=os.path.join(FIGURES_DIR, 'addresser-chronogram.pdf'))

    # Routing
    test_gate(out=os.path.join(FIGURES_DIR, 'gate-chronogram.pdf'))
    test_toggle(out=os.path.join(FIGURES_DIR, 'toggle-chronogram.pdf'))
    test_linked_router(out=os.path.join(FIGURES_DIR, 'linked-router-chronogram.pdf'))
    test_race_router(out=os.path.join(FIGURES_DIR, 'race-router-chronogram.pdf'))

    # Memory
    test_constant()
    test_boolean(out=os.path.join(FIGURES_DIR, 'boolean-chronogram.pdf'))
    test_timer(out=os.path.join(FIGURES_DIR, 'timer-chronogram.pdf'))
    test_volatile(out=os.path.join(FIGURES_DIR, 'volatile-memory-chronogram.pdf'))
    test_persistent(out=os.path.join(FIGURES_DIR, 'persistent-memory-chronogram.pdf'))
    test_parallel(out=os.path.join(FIGURES_DIR, 'parallel-memory-chronogram.pdf'))
    test_sequential(out=os.path.join(FIGURES_DIR, 'sequential-memory-chronogram.pdf'))
    test_chained(out=os.path.join(FIGURES_DIR, 'chained-memory-chronogram.pdf'))
    test_superimposed(out=os.path.join(FIGURES_DIR, 'superimposed-memory-chronogram.pdf'))

    test_spatial_memory(out=os.path.join(FIGURES_DIR, 'spatial-memory-chronogram.pdf'))
    test_addressable_memory(out=os.path.join(FIGURES_DIR, 'addressable-memory-chronogram.pdf'))
    test_linked_list(out=os.path.join(FIGURES_DIR, 'linked-list-chronogram.pdf'))

    # Algorithms
    test_sorting_v1(out=os.path.join(FIGURES_DIR, 'sorting-chronogram.v1.pdf'))
    test_sorting(out=os.path.join(FIGURES_DIR, 'sorting-chronogram.pdf'))
    test_searching(out=os.path.join(FIGURES_DIR, 'searching-chronogram.pdf'))

    # Misc
    test_busy_beaver()
