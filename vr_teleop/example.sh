#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Please provide an argument (peg_insertion, insert_cable, etc.)"
  exit 1
fi

# panda 
if [ "$1" == "peg_insertion" ]; then
  python3 tele.py --env configs/examples/maniskill_panda/peg_insertion_side.yaml --tele configs/agents/panda/tele_right.yaml
elif [ "$1" == "peg_insertion_rel" ]; then
  python3 tele.py --env configs/examples/maniskill_panda/peg_insertion_side.yaml --tele configs/agents/panda/tele_right_rel.yaml

elif [ "$1" == "insert_charger" ]; then
  python3 tele.py --env configs/examples/maniskill_panda/plug_charger.yaml --tele configs/agents/panda/tele_right.yaml
elif [ "$1" == "pick_cube" ]; then 
  python3 tele.py --env configs/examples/maniskill_panda/pick_cube.yaml --tele configs/agents/panda/tele_right.yaml
elif [ "$1" == "place_sphere" ]; then 
  python3 tele.py --env configs/examples/maniskill_panda/place_sphere.yaml --tele configs/agents/panda/tele_right.yaml
elif [ "$1" == "pull_cube" ]; then 
  python3 tele.py --env configs/examples/maniskill_panda/pull_cube_tool.yaml --tele configs/agents/panda/tele_right.yaml
elif [ "$1" == "push_cube" ]; then 
  python3 tele.py --env configs/examples/maniskill_panda/push_cube.yaml --tele configs/agents/panda/tele_right.yaml
elif [ "$1" == "assemble_kits" ]; then 
  python3 tele.py --env configs/examples/maniskill_xarm6/assembling_kits.yaml --tele configs/agents/xarm6/tele_right.yaml
elif [ "$1" == "lift_peg" ]; then 
  python3 tele.py --env configs/examples/maniskill_xarm6/lift_peg_upright.yaml --tele configs/agents/xarm6/tele_right.yaml
elif [ "$1" == "poke_cube" ]; then 
  python3 tele.py --env configs/examples/maniskill_xarm6/poke_cube.yaml --tele configs/agents/xarm6/tele_right.yaml
elif [ "$1" == "stack_cube" ]; then 
  python3 tele.py --env configs/examples/maniskill_xarm6/stack_cube.yaml --tele configs/agents/xarm6/tele_right.yaml

# fetch 
# elif [ "$1" == "fetch" ]; then 
#   python3 tele.py --env configs/examples/maniskill_fetch/robocasa_kitchen.yaml --tele configs/agents/fetch/tele_right.yaml
# # floating panda
# elif [ "$1" == "panda_gripper" ]; then 
#   python3 tele.py --env configs/examples/maniskill_floating_panda/robocasa_kitchen.yaml --tele configs/agents/floating_panda/tele_right.yaml
# panda stick
elif [ "$1" == "panda_stick" ]; then 
  python3 tele.py --env configs/examples/maniskill_panda_stick/push_t.yaml --tele configs/agents/panda_stick/tele_right.yaml
# allegro 
# elif [ "$1" == "allegro" ]; then 
#   python tele.py --env configs/examples/maniskill_allegro/open_notebook.yaml --tele configs/agents/floating_allegro_hand/tele_right.yaml
# elif [ "$1" == "rotate_cube" ]; then 
#   python tele.py --env configs/examples/maniskill_allegro/rotate_single_object.yaml --tele configs/agents/floating_allegro_hand/tele_right.yaml
# floating ability hand
elif [ "$1" == "insert_flower" ]; then 
  python3 tele.py --env configs/examples/maniskill_floating_ability/insert_flower.yaml --tele configs/agents/floating_ability_hand/tele_right.yaml
else
  echo "Invalid argument. "
  exit 1
fi