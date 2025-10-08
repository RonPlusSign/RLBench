source setup.sh
conda activate openpi

NUM_EPISODES=3

# python test_openpi.py "put_rubbish_in_bin" $NUM_EPISODES 2>&1 | tee runs/openpi_test/output_put_rubbish_in_bin.txt
python test_openpi.py "put_books_on_bookshelf" $NUM_EPISODES 2>&1 | tee runs/openpi_test/output_put_books_on_bookshelf.txt
# python test_openpi.py "empty_container" $NUM_EPISODES 2>&1 | tee runs/openpi_test/output_empty_container.txt
