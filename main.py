import praw
import csv
reddit = praw.Reddit(
    client_id="XxtFsMTYONOz3opDovuo6A",
    client_secret="UQsuqpFzrDUN2odyEraeIkCriepecA",
    user_agent="my user agent",
)

ask_science = reddit.subreddit("askscience")


with open('prompts.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    submissions = ask_science.top(time_filter="all", limit=2000)

    for i, submission in enumerate(submissions):
        print(submission.title)
        submission.comment_sort = "top"
        print("sorted")
        for c in submission.comments:
            if not c.stickied:
                print("found")
                top_comment = c
                break
            else:
                print("stickied")
        writer.writerow([i, submission.title, top_comment.body])
