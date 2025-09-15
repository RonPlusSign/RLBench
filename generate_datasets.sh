source setup.sh

# Tasks: PutRubbishInBin, PutBooksOnBookshelf, EmptyContainer

# Absolute actions, Euler rotation representation
# python generate_dataset_rlds.py --task="PutRubbishInBin" --num_episodes=100 --action_repr="euler" 2>&1 | tee datasets/euler_absolute/output_put_rubbish_in_bin_absolute_euler.txt
# python generate_dataset_rlds.py --task="PutBooksOnBookshelf" --num_episodes=100 --action_repr="euler" 2>&1 | tee datasets/euler_absolute/output_put_books_on_bookshelf_absolute_euler.txt
# python generate_dataset_rlds.py --task="EmptyContainer" --num_episodes=100 --action_repr="euler" 2>&1 | tee datasets/euler_absolute/output_empty_container_absolute_euler.txt

# Absolute actions, quaternion rotation representation
# python generate_dataset_rlds.py --task="PutRubbishInBin" --num_episodes=100 --action_repr="quat" 2>&1 | tee datasets/quat_absolute/output_put_rubbish_in_bin_absolute_quat.txt
# python generate_dataset_rlds.py --task="PutBooksOnBookshelf" --num_episodes=100 --action_repr="quat" 2>&1 | tee datasets/quat_absolute/output_put_books_on_bookshelf_absolute_quat.txt
# python generate_dataset_rlds.py --task="EmptyContainer" --num_episodes=100 --action_repr="quat" 2>&1 | tee datasets/quat_absolute/output_empty_container_absolute_quat.txt

# Relative actions, Euler rotation representation
# python generate_dataset_rlds.py --task="PutRubbishInBin" --num_episodes=100 --absolute_actions=false --action_repr="euler" 2>&1 | tee datasets/euler_relative/output_put_rubbish_in_bin_relative_euler.txt
# python generate_dataset_rlds.py --task="PutBooksOnBookshelf" --num_episodes=100 --absolute_actions=false --action_repr="euler" 2>&1 | tee datasets/euler_relative/output_put_books_on_bookshelf_relative_euler.txt
# python generate_dataset_rlds.py --task="EmptyContainer" --num_episodes=100 --absolute_actions=false --action_repr="euler" 2>&1 | tee datasets/euler_relative/output_empty_container_relative_euler.txt

# Relative actions, quaternion rotation representation
# python generate_dataset_rlds.py --task="PutRubbishInBin" --num_episodes=100 --absolute_actions=false --action_repr="quat" 2>&1 | tee datasets/quat_relative/output_put_rubbish_in_bin_relative_quat.txt
# python generate_dataset_rlds.py --task="PutBooksOnBookshelf" --num_episodes=100 --absolute_actions=false --action_repr="quat" 2>&1 | tee datasets/quat_relative/output_put_books_on_bookshelf_relative_quat.txt
# python generate_dataset_rlds.py --task="EmptyContainer" --num_episodes=100 --absolute_actions=false --action_repr="quat" 2>&1 | tee datasets/quat_relative/output_empty_container_relative_quat.txt
