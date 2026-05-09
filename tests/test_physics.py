from __future__ import annotations

import unittest

from exact.physics import solve_physics_question


class PhysicsTests(unittest.TestCase):
    def test_capacitor_energy(self) -> None:
        result = solve_physics_question(
            "Calculate the energy stored in a capacitor when C = 0.1 F and U = 30 V."
        )
        self.assertEqual(result["answer"], "45")
        self.assertEqual(result["unit"], "J")

    def test_ohms_law_current(self) -> None:
        result = solve_physics_question("Find the current when U = 12 V and R = 6 ohm.")
        self.assertEqual(result["answer"], "2")
        self.assertEqual(result["unit"], "A")


if __name__ == "__main__":
    unittest.main()

