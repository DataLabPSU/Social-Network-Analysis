Study 2 data captured 4/4/19

Post processed data for node-node edgelist, node-group membership, node-credibility score list, node-impressions list, node-likes list.

Format for each data file:

Edgelist
format: <userid1> <following1>
	    <userid1> <following2>
        <userid2> <following1>

Group label list
format: <userid> <label1> <label2> <label3>

Impressions list
format: <userid> <num_following> <num_followers> <num_likes> <num_shares>
        
Credibility score list
format: <userid> <credibility_score> 

Likes list
format: <userid> <real_likes> <fake_likes>
