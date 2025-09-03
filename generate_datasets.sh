source setup.sh

# Tasks: PutRubbishInBin, PutBooksOnBookshelf, EmptyContainer


# Absolute actions, quaternion rotation representation (defaults)
# python generate_dataset.py --task="PutRubbishInBin" --num_episodes=100 --save_path="datasets/absolute EE, quaternion repr" 2>&1 | tee logs/output_put_rubbish_in_bin_absolute_quat.txt
# python generate_dataset.py --task="PutBooksOnBookshelf" --num_episodes=50 --save_path="datasets/absolute EE, quaternion repr" 2>&1 | tee logs/output_put_books_on_bookshelf_absolute_quat.txt
# python generate_dataset.py --task="EmptyContainer" --num_episodes=50 --save_path="datasets/absolute EE, quaternion repr" 2>&1 | tee logs/output_empty_container_absolute_quat.txt

# Relative actions, quaternion rotation representation
# python generate_dataset.py --task="PutRubbishInBin" --num_episodes=100 --absolute_actions=false --save_path="datasets/relative EE, quaternion repr" 2>&1 | tee logs/output_put_rubbish_in_bin_relative_quat.txt
# python generate_dataset.py --task="PutBooksOnBookshelf" --num_episodes=50 --absolute_actions=false --save_path="datasets/relative EE, quaternion repr" 2>&1 | tee logs/output_put_books_on_bookshelf_relative_quat.txt
# python generate_dataset.py --task="EmptyContainer" --num_episodes=50 --absolute_actions=false --save_path="datasets/relative EE, quaternion repr" 2>&1 | tee logs/output_empty_container_relative_quat.txt

# Absolute actions, Euler rotation representation
# python generate_dataset.py --task="PutRubbishInBin" --num_episodes=100 --action_repr="euler" --save_path="datasets/absolute EE, euler repr" 2>&1 | tee logs/output_put_rubbish_in_bin_absolute_euler.txt
# python generate_dataset.py --task="PutBooksOnBookshelf" --num_episodes=50 --action_repr="euler" --save_path="datasets/absolute EE, euler repr" 2>&1 | tee logs/output_put_books_on_bookshelf_absolute_euler.txt
# python generate_dataset.py --task="EmptyContainer" --num_episodes=50 --action_repr="euler" --save_path="datasets/absolute EE, euler repr" 2>&1 | tee logs/output_empty_container_absolute_euler.txt

# Relative actions, Euler rotation representation
# python generate_dataset.py --task="PutRubbishInBin" --num_episodes=100 --absolute_actions=false --action_repr="euler" --save_path="datasets/relative EE, euler repr" 2>&1 | tee logs/output_put_rubbish_in_bin_relative_euler.txt
# python generate_dataset.py --task="PutBooksOnBookshelf" --num_episodes=50 --absolute_actions=false --action_repr="euler" --save_path="datasets/relative EE, euler repr" 2>&1 | tee logs/output_put_books_on_bookshelf_relative_euler.txt
# python generate_dataset.py --task="EmptyContainer" --num_episodes=50 --absolute_actions=false --action_repr="euler" --save_path="datasets/relative EE, euler repr" 2>&1 | tee logs/output_empty_container_relative_euler.txt
