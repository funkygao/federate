---- MODULE SimpleCounter ----
EXTENDS Integers

VARIABLE counter

Init == counter = 0

Increment == counter' = counter + 1

Next == Increment

CounterIsNatural == counter >= 0

Spec == Init /\ [][Next]_counter

====
