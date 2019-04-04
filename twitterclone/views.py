from django.shortcuts import render, redirect, render_to_response, get_object_or_404
from django.template import RequestContext
from .models import Post, Comment, Share, Profile, Message
from . import forms
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.utils.timezone import make_aware
from django.conf import settings
from django.template.defaulttags import register

import datetime, time
import random, os, json


def updateCredibilityScore(request):
	d = {}
	for user in User.objects.all():
		temp = 0.1 * (user.profile.real + 1) / (user.profile.real + user.profile.fake + 2)
		temp2 = 1 - 0.1
		temp3 = 1 - (user.profile.real + 1) / (user.profile.real + user.profile.fake + 2)

		d[user.username] = temp / (temp + temp2 * temp3)
	 	
	 	# skip if credibilityscore doesn't change
		if user.profile.credibilityscore == d[user.username]:
	 		continue

		temp = User.objects.get(username=user.username)
		temp.profile.credibilityscore = d[user.username]
		temp.save()

updateLikeRequested = False
def updatelike(request):
	#block spam- one request at a time
	global updateLikeRequested
	if updateLikeRequested:
		data = {}
		updateLikeRequested = False
		return JsonResponse(data)
	else:
		updateLikeRequested = True

	postid = request.GET.get('postid')
	user = User.objects.get(pk=request.user.id)
	post = Post.objects.get(pk=postid)
	try:
		if post.author != user and postid not in user.profile.liked:
			post.likes += 1
			# append video labels to user labels
			user.profile.labels = user.profile.labels + post.videolabels + "|"
			post.updated = make_aware(datetime.datetime.now())

			if post.real == 0:
				user.profile.fake = user.profile.fake + 1
			else:
				user.profile.real = user.profile.real + 1
			user.profile.liked = user.profile.liked + " " + postid
			post.save()
			user.save()
		elif post.author != user and postid in user.profile.liked:
			temp = user.profile.liked.split(" ")
			temp.remove(postid)
			user.profile.liked = ' '.join(temp)

			# remove first instance of post videolabels 
			user.profile.labels = user.profile.labels.replace(post.videolabels + "|", "", 1)
			
			post.likes -= 1
			if post.real == 0:
				user.profile.fake = user.profile.fake - 1
			else:
				user.profile.real = user.profile.real - 1
			post.save()
			user.save()
	except:
		pass

	data = {
		'likes': post.likes
	}
	
	# update flag to allow next request
	updateLikeRequested = False

	return JsonResponse(data)

def sharepost(request):
	try:
		postid = request.GET.get('postid')
		sharecomment = request.GET.get('sharecomment')
		if not postid:
			pass

		post = Post.objects.get(id=postid)
		if request.user != post.author and request.user.username not in post.text:
			if sharecomment == '':
				post.text = post.text + '\n' + 'Post shared by ' + request.user.username
			else:
				post.text = post.text + '\n' + 'Post shared by ' + request.user.username + ' with comment ' + sharecomment
			
			newshare = Share.objects.create(shared=request.user)
			newshare.postid = postid 
			try:
			 	newshare.comment = sharecomment
			except Exception as e:
			 	print(str(e))
			 	pass

			# newshare.save()
			user = request.user
			# add video labels to user for sharing
			user.profile.labels = user.profile.labels + post.videolabels + "|" 
			print(user.profile.labels)
			if post.real == 0:
				user.profile.fake = user.profile.fake + 1
			else:
				user.profile.real = user.profile.real + 1
			user.save()
			post.save()
			for user in User.objects.all():
				if request.user.username in user.profile.following.split(" "):
					user.profile.notifications = user.profile.notifications + 1
					user.profile.notificationsString += "Post shared by " + str(request.user) + " at " + str(datetime.datetime.now()) + "|"
					user.save()

	except Exception as e:
		pass

	data = {
		'posttext': post.text
	}
	
	return JsonResponse(data)

def likepost(request, userid, post):
	user = User.objects.get(pk=userid)
	postid = str(post.id)

	if postid not in user.profile.liked:
		post.likes += 1
		# append video labels to user labels
		user.profile.labels = user.profile.labels + post.videolabels + "|"
		post.updated = make_aware(datetime.datetime.now())

		if post.real == 0:
			user.profile.fake = user.profile.fake + 1
		else:
			user.profile.real = user.profile.real + 1
		user.profile.liked = user.profile.liked + " " + postid
		print('liked')
		post.save()
		user.save()
	elif postid in user.profile.liked:
		temp = user.profile.liked.split(" ")
		temp.remove(postid)
		user.profile.liked = ' '.join(temp)

		# remove first instance of post videolabels 
		user.profile.labels = user.profile.labels.replace(post.videolabels + "|", "", 1)
		
		post.likes -= 1
		if post.real == 0:
			user.profile.fake = user.profile.fake - 1
		else:
			user.profile.real = user.profile.real - 1
		post.save()
		user.save()
	
	return True

def createusers(request):
	for i in range(0,501):
		user = User.objects.create_user(username='user' + str(i),
                                 email='',
                                 password='glass' + str(i))

	return render(request,'twitterclone/agree.html')

def loadtest(request):
	users = User.objects.all()
	posts = Post.objects.all()
	for user in users:
		for post in posts:
			print('likepost ' + str(user.id) + ' ' + str(post.id))
			likepost(request, user.id, post)

	return render(request,'twitterclone/agree.html')

def processdata(request):
	users = User.objects.all()
	edgelist_format = '{userid} {followingid}'
	labellist_format = '{userid} {label_x}'
	grouplist = {'Audioless': 0,
					'International News' : 1,
					'Domestic News' : 2,
					'Political' : 3,
					'Healthcare' : 4,
					'Random' : 5,
					'Face-altered' : 6,
					'Fake' : 7,
					'Advertisement' : 8,
					'Sports' : 9,
					'Movie' : 10,
					'Education' : 11,
	        'Business': 12,
	        'Faked': 7}

	pdatadir = 'postdata/'
	timenow =  datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
	pedgelist_file = pdatadir + 'edgelist_' + timenow + '.txt'
	plabellist_file = pdatadir + 'labellist_' + timenow + '.txt'
	pimpressions_file = pdatadir + 'impressions_' + timenow + '.txt'
	pcredibility_file = pdatadir + 'credibility_' + timenow	+ '.txt'

	# edgelist
	# format: <userid> <following1>
	#		  <userid> <following2>
	with open(pedgelist_file, 'w') as edgefile:
		for user in users:
			try:
				following = user.profile.following
				userid = (str)(user.id)
				following = following.split(' ')
				for follower in following:
					if follower:
						followerid = (str)(users.get(username=follower).id)
						edgefile.write(userid + ' ' + followerid + '\n')
			except Exception as e:
			 	print(str(e))
			 	pass	
			#print(userid + ' ' + following)
			#print(userid + ' ' + labels)
			#print('\n\n')

	# labellist
	# format: <userid> <label1> <label2> <label3>
	with open(plabellist_file, 'w') as labelfile:
		for user in users:
			try:
				labels = user.profile.labels
				userid = str(user.id)
				if labels == "":
					labelfile.write(userid + '\n')
					continue

				labels = labels.replace("|", ",")
				labels = labels.split(",")
				groupids = []
				for label in labels:
					if label and grouplist[label.strip()]:
						groupids.append(grouplist[label.strip()])
				gids = list(set(groupids))
				labelline = userid + " " + " ".join(str(x) for x in gids)
				labelfile.write(labelline + '\n')
			except Exception as e:
			 	print(str(e))
			 	pass

	# followers, likes, shares list
	# format: <userid> <following> <followers> <numlikes> <numshares>
	with open(pimpressions_file, 'w') as impressionsfile:
		for user in users:
			try:
				userid = str(user.id)
				following = len(user.profile.following.split(" "))
				followeenum = 0
				for i in users:
					if user.username in i.profile.following.split(" "):
						followeenum += 1
				numliked = len(user.profile.liked.split(" "))
				numshares = Share.objects.filter(shared=user.id).count()
				impressionsline = userid + ' ' + str(following) + ' ' + str(followeenum) + ' ' + str(numliked) + ' ' + str(numshares)
				impressionsfile.write(impressionsline + '\n')
			except Exception as e:
			 	print(str(e))
			 	pass

	updateCredibilityScore(request)
	with open(pcredibility_file, 'w') as credibilityfile:
		for user in users:
			try:
				credibilityline = str(user.id) + ' ' + str(user.profile.credibilityscore)
				credibilityfile.write(credibilityline + '\n')
			except Exception as e:
				print(str(e))
				pass

	return render(request,'twitterclone/agree.html')

# resets all user data for all users
def resetuserdata(request):
	# reset all user data except root and admin
	if request.user == User.objects.get(username='admin'):
		for user in User.objects.all():
			user.profile.following = ''
			user.profile.labels = ''
			user.profile.real = 0
			user.profile.fake = 0
			user.profile.credibilityscore = 0.1
			user.save()
	
	return redirect('home')


# purges all posts, shares, comments
def deleteposts(request):
	# delete all posts except for videos
	if request.user == User.objects.get(username='admin'):
		Post.objects.filter(video=None).delete()
		Share.objects.filter().delete()
	return redirect('home')
	
# resets all videos data
def resetvideos(request):
	if request.user == User.objects.get(username='admin'):
		for post in Post.objects.all():
			post.likes = 0
			post.created_date = make_aware(datetime.datetime.now())
			post.updated = make_aware(datetime.datetime.now())
			post.text = ''
			post.save()
	return redirect('home')

# resets all users except admin
def deleteusers(request):
	# delete all users except root and admin
	if request.user == User.objects.get(username='admin'):
		for user in User.objects.all():
			if user.username in ['admin']:
				user.profile.following = ''
				user.profile.labels = ''
				user.profile.real = 0
				user.profile.fake = 0
				user.profile.credibilityscore = 0.1
				user.save()
				continue
			user.delete()
		
	return redirect('home')

# save youtube video ids to db
def addvideoids(request):
	with open('media/videos/video_ids.json') as f:
		data = json.load(f)
	for post in Post.objects.all():
		try:
			vidname = post.videoname
			if vidname != '':
				isFake = 'Fake' in vidname
				vidnum = vidname[:-4].split('_')[1]

				# real videos range from 31-60
				if not isFake:
					vidnum = str(int(vidnum) + 30)

				if data[vidnum]:
					post.video = data[vidnum]

				post.save()
			else:
				post.video = ''
				post.save()
		except:
			pass
	return render(request,'twitterclone/agree.html')

# adds videos
def addvideos(request):
	# add videos script
	if request.user == User.objects.get(username='root2'):  
		os.chdir('media/videos/')
		vids = os.listdir()
		
		# shuffle order of videos
		random.shuffle(vids)

		# video labels formatting:
		# <video_id>. <label1>, <label2>...
		# ex: 3. Audioless, Advertisement, Face-altered
		#fake video labels
		f = open("Fake_Video_Labels.txt", "r")
		fake_labels = f.readlines()
		f.close()

		#real video labels
		t = open("True_Video_Labels.txt", "r")
		true_labels = t.readlines()
		t.close()

		# allowed video extensions
		allowed_ext = ['mp4', 'mov']

		for i in vids:
			if i[-3:] in allowed_ext:

				# determine if video is fake or not from filename
				isFake = 'Fake' in i

				# create new post for admin
				user = User.objects.get(username='admin')
				temp = Post.objects.create(author=user)
				temp.title = 'Video'
				temp.video = ''

				# fetch video id
				videoid = i[:-3].split('_')[-1]

				# determine video label from label list
				labels = fake_labels if isFake else true_labels
				for label in labels:
					if videoid in label:
						videolabel = label
						break
				
				# add video label
				temp.videolabels = videolabel.split('.')[-1].strip()  

				# add subtitles if provided
				if (i[:-3]+"en.vtt") in os.listdir():
				   temp.subtitles = i[:-3]+"en.vtt"

				# based on isFake, 0 for fake 1 for real
				temp.real = 0 if isFake else 1 
				temp.save()
	return redirect('home')

def home(request):
	'''
	for i in range(5):
			user = User.objects.get(username='vic')
			temp = Post.objects.create(author=user)
			temp.title = 'Fake'
			temp.text = 'This is a fake post'
			temp.real = 0
			temp.save()
   '''
 # if datetime.datetime.now().hour > 12:
	#    return render(request,'twitterclone/end.html')
	d = {}
	print(request.POST)
	followerscount = {}

	if request.user.is_authenticated:		
		if request.method == 'POST':
			try:
				follow = request.POST['follows']
				# don't allow users to follow admin
				if not follow or follow == "admin":
					pass
				user = request.user
				user.profile.following = user.profile.following + " " + follow
				user.save()
			except:
				pass

			try:
				user = User.objects.get(pk=request.user.id)
				temp = set(user.profile.following.split(" "))
			 
				temp.remove(request.POST['follow'])
				
				user.profile.following = ' '.join(temp)
				user.save()
			except:
				pass
			
			try:
				comment = request.POST['placeholder']
				if not comment:
					pass

				newcomment = Comment.objects.create(author=request.user)
				newcomment.post = Post.objects.get(pk=request.POST['submit'])
				newcomment.text = comment
				temp = Post.objects.get(pk=request.POST['submit'])
				temp.updated = make_aware(datetime.datetime.now())
				temp.save()
				newcomment.save()
				for user in User.objects.all():
					if request.user.username in user.profile.following.split(" "):
						user.profile.notifications = user.profile.notifications + 1
						user.save()
			except:
				pass
		user = User.objects.get(pk=request.user.id)

		if request.user.username == 'admin':
			postlist = []
		else:
			posts = Post.objects.filter(author=request.user)
			postlist = list(posts)
		# shares = Share.objects.filter(shared=request.user)
		
		# # append retweets to postlist
		# for z in shares:
		# 	content = Post.objects.get(id=z.postid)
		# 	if content.author.username == 'admin':
		# 		content.author.username = 'Retweeted by ' + z.shared.username
		# 	else:
		# 		content.author.username = content.author.username + ' Retweeted by ' + z.shared.username
		# 	content.created_date = z.date
		# 	content.sharecomment = z.comment

		# 	postlist.append(content)

		# following
		# if user.profile.following != "":
		# 	for i in set(user.profile.following.split(" ")):
		# 		if i != "":

		# 			#skip appending admin posts
		# 			if i == "admin":
		# 				continue

		# 			following = User.objects.get(username=i)

		# 			otherposts = Post.objects.filter(author=following)
		# 			posts = posts | otherposts
		# 			postlist.extend(list(otherposts))
		# 			sharedposts = Share.objects.filter(shared=following)
		# 			for z in sharedposts:
		# 				content = Post.objects.get(id=z.postid)
		# 				if content.author.username == "admin":
		# 					content.author.username =  'Shared by ' + z.shared.username
		# 				else:
		# 					content.author.username = content.author.username + ' Shared by ' + z.shared.username
		# 				content.created_date = z.date
		# 				content.sharecomment = z.comment
		# 				postlist.append(content)
		notificationsString = request.user.profile.notificationsString.split("|")
		userlist = User.objects.exclude(pk=request.user.id)
		finaloutput = []
		temp = (Profile.objects.all().order_by('-credibilityscore'))[:15]
		dictionary = {}
		#print(postlist)
		postlist.sort(key=lambda r:r.updated,reverse=True) 
		#print(postlist)

		# start_time = time.time()
		# updateCredibilityScore(request)
		# print(time.time() - start_time)
		# start_time = time.time()

		for i in User.objects.all():
			# skip admins
			if i.username == 'admin':
				continue

			for z in set(i.profile.following.split(" ")):
				try:
				   dictionary[z] += 1
				except:
				   dictionary[z] = 1
		# print(time.time() - start_time)
		# start_time = time.time()
		for i in temp:
			#skip admin
			if i.user.username == 'admin':
				continue

			if i.user != request.user and i.user.username not in request.user.profile.following.split(" "):
				 
				 try:
				   finaloutput.append([i.user,dictionary[i.user.username]])
				 except:
				   finaloutput.append([i.user,0])   
		# print(time.time() - start_time)
		# start_time = time.time()
		following = []
		for i in set(user.profile.following.split(" ")[1:]):
			following.append(User.objects.get(username=i))
		followeenum = 0
		for i in User.objects.all():
			if request.user.username in i.profile.following.split(" "):
				followeenum += 1
		# print(time.time() - start_time)
		# start_time = time.time()
		#print(finaloutput)
		for i in finaloutput:
			if i[0] in following:
				finaloutput.remove([i[0],i[1]])
		# print(time.time() - start_time)
		start_time = time.time()
		mainvids = Post.objects.filter(author=User.objects.get(username='admin'))
		for mainvid in mainvids:
			mainvid.comments = Comment.objects.filter(post=mainvid.id).order_by('-created_date')[:30]
		# print(time.time()-start_time)

		context = {
			'mainvids': mainvids,
			'posts': postlist,
			'FOLLOWING': following,
			'currentuser': request.user,
			'notifications': notificationsString,
			'users': userlist,
			'numfollowers': finaloutput,
			'curfollowersnum': len(following),
			'curfolloweesnum': followeenum,
			'image': request.user.profile.imagename
		}
		
		return render(request, "twitterclone/home.html", context)
	else:
		# send to consent form if not logged in
		return render(request, "twitterclone/agree.html")
		#return redirect('login')

def signup(request):
	if request.method == 'POST':
		form = forms.UserCreationForm(request.POST)
		if form.is_valid():
			form.save()
			username = form.cleaned_data.get('username')
			raw_password = form.cleaned_data.get('password1')
			user = authenticate(username=username, password=raw_password)
			login(request, user)
			return redirect('home')
	else:
		if request.GET:
			test_form = forms.UserCreationForm(data=request.GET)
			test_form.is_valid()
			form = forms.UserCreationForm(initial=test_form.cleaned_data)
		else:
			form = forms.UserCreationForm()
	return render(request, 'twitterclone/signup.html', {'form': form})

def instructions(request):
	return render(request,'twitterclone/instructions.html')

def survey(request):
   
	temp = User.objects.get(pk=request.user.id)
	context = {
	'user': temp,
	}
	if request.method == 'POST':
		form = forms.SurveyForm(request.POST)
		if form.is_valid():
			instance = form.save(commit=False)
			instance.author = request.user
			instance.save()

			return render(request,'twitterclone/final.html',context)
	else: 
		form = forms.SurveyForm()
	return render(request,'twitterclone/survey.html',{'form':form})

def addpost(request):
	if request.method == 'POST':
		form = forms.PostForm(request.POST)
		if form.is_valid():
			instance = form.save(commit=False)
			instance.author = request.user
			instance.save()
			return redirect('home')
	else:
		form = forms.PostForm()

	return render(request, 'twitterclone/create.html', {'form': form})


def notification(request):
	temp = request.user.profile.notificationsString
	num = request.user.profile.notifications
	context = {
		'notifications': temp.split("|"),
		'num': num,
	}
	return render(request, 'twitterclone/notification.html', context)


def testfollow(request):
	if request.method == 'POST':
		user = User.objects.get(pk=request.user.id)
		user.profile.following = user.profile.following + " " + request.POST['follow']
		user.save()

	userlist = User.objects.exclude(pk=request.user.id)
	context = {
		'users': userlist
	}
	return render(request, 'twitterclone/follow.html', context)

def pick(request):
	if request.method == 'POST':
		user = User.objects.get(pk=request.user.id)
		user.profile.imagename = request.POST['hidden']
		user.save()
		return redirect('home')
	os.chdir(settings.MEDIA_ROOT + '/images/')
	images = [i for i in os.listdir()[:15]]
	images = [i for i in images if i[-3:] != 'jpg']
	context = {
		'images': images,
		'user': request.user,
	}
	print(context)
	return render(request, 'twitterclone/settings.html', context)

def final(request):
	temp = User.objects.get(pk=request.user.id)
	context = {
	'user': temp,
	}
	return render(request, 'twitterclone/final.html',context)

def pm(request):
	if request.method == 'POST':
		temp = Message.objects.create(text=request.POST['message'])
		temp.recipient = request.POST['to']
	d = {}
	for i in Message.objects.all():
		if i.recipient == request.user.username or i.author == request.user:
			if i.recipient == request.user.username:
				try:
					d[i.author.username].append(i.text)
				except:
					d[i.author.username] = [i.text]
			else:
				try:
					d[i.recipient].append(i.text)
				except:
					d[i.recipient] = [i.text]
	d['vic'] = ['hello there', 'hi']
	context = {
		'messages': d
	}
	print(d)
	return render(request, 'twitterclone/pm.html', context)
