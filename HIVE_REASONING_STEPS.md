# Reasoning steps

Step zero is an immutable `INITIAL` snapshot. Each requested step performs query energy, shake, resonance propagation, gravity/activation/retention updates, movement, decay, eviction and settle ticks, then writes an immutable `AFTER_SETTLE` snapshot. Later steps begin with the previous final state and may stop early on convergence.

