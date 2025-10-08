source setup.sh
export MESA_LOADER_DRIVER_OVERRIDE=swrast

NUM_EPISODES=10

python test_openvla.py "put_rubbish_in_bin" $NUM_EPISODES 2>&1 | tee runs/openvla_test/output_put_rubbish_in_bin.txt
# python test_openvla.py "put_books_on_bookshelf" $NUM_EPISODES 2>&1 | tee runs/openvla_test/output_put_books_on_bookshelf.txt
# python test_openvla.py "empty_container" $NUM_EPISODES 2>&1 | tee runs/openvla_test/output_empty_container.txt
