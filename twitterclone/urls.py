from django.conf.urls import url
from . import views
from django.contrib import admin
from django.urls import include
from django.urls import path
urlpatterns=[
path('signup/',views.signup,name='signup'),
path('',views.home,name='home'),
path('create/',views.addpost,name='addpost'),
path('notification/', views.notification, name='notification'),
path('settings/',views.pick,name='settings'),
path('pm/',views.pm,name='pm'),
path('instructions/',views.instructions,name='instructions'),
path('final/',views.survey,name='survey')
# path('follow/',views.testfollow,name='testfollow')
]



