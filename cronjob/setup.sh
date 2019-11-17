# set up necessary files
mkdir -p ~/482-cron-job
cp ./run-job.sh ~/482-cron-job/run-job.sh
cp ../db.py ~/482-cron-job/job.py
cp ../credentials.py ~/482-cron-job/credentials.py
rm -f ~/482-cron-job/output.txt

# add entry to cron table
echo "Next, copy the command on the following line:"
echo "*/15 * * * * ~/482-cron-job/run-job.sh >> ~/482-cron-job/output.txt 2>&1"
echo "Then, paste it in the file that opens when you run the command: crontab -e"
