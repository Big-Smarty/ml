"""Course-authored maintenance fixture v1, not observations from a real plant.
Re-run only for an intentional new dataset version; never tune after final evaluation.
Python stdlib, fixed RNG, rounded stored measurements. No downloads.
"""
from pathlib import Path
import math
import random
rng = random.Random(20260909)
lines = ['id,machine,day,temperature,vibration,load,regime,failure,repair_after,probability']
for machine in range(24):
    offset = rng.uniform(-0.6, 0.6)
    for day in range(12):
        regime = ('idle', 'loaded')[(machine + day // 2) % 2]
        if machine >= 18 and day >= 9 and machine % 2 == 0:
            regime = 'turbo'
        load = (0.3 if regime == 'idle' else 0.8) + rng.uniform(-0.16, 0.16)
        wear = (machine % 6) / 5 + day * 0.075 + offset
        vibration = 0.6 + 1.5 * wear + 0.65 * load + rng.uniform(-0.4, 0.4)
        temperature = 48 + 8 * wear + 12 * load + rng.uniform(-3, 3)
        # Correlated readings share wear. Future temperature bias mimics a sensor replacement.
        if day >= 9:
            temperature += 5
        latent = -3.2 + 1.65 * wear + 1.0 * load + 1.7 * (wear > 0.8 and load > 0.65)
        latent += 0.55 * (day >= 9) + rng.uniform(-0.65, 0.65)
        probability = 1 / (1 + math.exp(-latent))
        failure = int(rng.random() < probability)
        # The inspection outcome is available day+1 and is an explicit target leak.
        repair_after = 'replaced' if failure else 'none'
        values = [f'{temperature:.3f}', f'{vibration:.3f}', f'{load:.3f}']
        if (machine * 7 + day) % 11 == 0 or (failure and day % 5 == 0):
            values[0] = ''
        if (machine + 3 * day) % 17 == 0:
            values[1] = ''
        # Frozen rule-of-thumb risk, never fitted on any split; used in chapter 12 only.
        heuristic = 1 / (1 + math.exp(-(-4.5 + 1.5 * vibration + 1.2 * load)))
        lines.append(','.join(map(str, [machine * 12 + day, machine, day, *values, regime, failure, repair_after, f'{heuristic:.6f}'])))
Path(__file__).with_name('maintenance.csv').write_text('\n'.join(lines) + '\n')
