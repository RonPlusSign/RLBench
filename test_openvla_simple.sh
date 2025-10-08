source setup.sh
export MESA_LOADER_DRIVER_OVERRIDE=swrast

NUM_EPISODES=10

# Test without finetuning
# python test_openvla_simple.py --task_name="put_rubbish_in_bin" --n_episodes=$NUM_EPISODES --checkpoint="openvla/openvla-7b" 2>&1 | tee runs/openvla_simple_test/output_put_rubbish_in_bin_no_finetuning.txt
# python test_openvla_simple.py --task_name="put_books_on_bookshelf" --n_episodes=$NUM_EPISODES --checkpoint="openvla/openvla-7b" 2>&1 | tee runs/openvla_simple_test/output_put_books_on_bookshelf_no_finetuning.txt
# python test_openvla_simple.py --task_name="empty_container" --n_episodes=$NUM_EPISODES --checkpoint="openvla/openvla-7b" 2>&1 | tee runs/openvla_simple_test/output_empty_container_no_finetuning.txt

# Test with finetuning on put_rubbish_in_bin
# python test_openvla_simple.py --task_name="put_rubbish_in_bin" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+PutRubbishInBin_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_put_rubbish_in_bin_finetuned_on_bin.txt
# python test_openvla_simple.py --task_name="put_books_on_bookshelf" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+PutRubbishInBin_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_put_books_on_bookshelf_finetuned_on_bin.txt
# python test_openvla_simple.py --task_name="empty_container" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+PutRubbishInBin_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_empty_container_finetuned_on_bin.txt

# Test with finetuning on put_books_on_bookshelf
python test_openvla_simple.py --task_name="put_rubbish_in_bin" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+PutBooksOnBookshelf_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_put_rubbish_in_bin_finetuned_on_bookshelf.txt
python test_openvla_simple.py --task_name="put_books_on_bookshelf" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+PutBooksOnBookshelf_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_put_books_on_bookshelf_finetuned_on_bookshelf.txt
python test_openvla_simple.py --task_name="empty_container" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+PutBooksOnBookshelf_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_empty_container_finetuned_on_bookshelf.txt

# Test with finetuning on empty_container
python test_openvla_simple.py --task_name="put_rubbish_in_bin" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+EmptyContainer_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_put_rubbish_in_bin_finetuned_on_empty_container.txt
python test_openvla_simple.py --task_name="put_books_on_bookshelf" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+EmptyContainer_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_put_books_on_bookshelf_finetuned_on_empty_container.txt
python test_openvla_simple.py --task_name="empty_container" --n_episodes=$NUM_EPISODES --checkpoint="/home/adelli/openvla/checkpoints/openvla-7b+EmptyContainer_euler_relative+b10+lr-0.0005+lora-r32+dropout-0.0--image_aug" 2>&1 | tee runs/openvla_simple_test/output_empty_container_finetuned_on_empty_container.txt