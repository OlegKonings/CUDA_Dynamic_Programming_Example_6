
CUDA adaptation of the Top Coder Division I problem:

http://community.topcoder.com/stat?c=problem_statement&pm=6412&rd=9825&rm=&cr=2058177

分享您的知识，和其他人的工作不邀功！

Yes, another CUDA implementation of a 64 bit double precision probability dynamic programming problem. While not yet optimizes, still runs 19x-30x times faster than an optimized 3.9 Ghz CPU serial implementation. So far tests between the two implementations yield the exact same results, but not tested enough to verify there are no pathological cases.

If error checking in CUDA code was removed, and the reduction step optimized, at least 2 ms will be shaved off the GPU running time. For GPUs with a compute capability less than 3.5, use 32 bit floating point numbers for faster performance.

Apx iterations are (nDice+1)*((nDice+1)*(maxSide+1))*(maxSide+1)*2 + ((nDice+1)*(maxSide+1))
____
<table>
<tr>
    <th>nDice</th><th>maxSide</th><th>Apx iterations</th><th>CPU time</th><th>GPU time</th><th>CUDA Speedup</th>
</tr>

  <tr>
    <td>50</td><td>50</td><td>13,533,003</td><td> 96 ms</td><td>  5 ms</td><td> 19.2x</td>
  </tr>
  <tr>
    <td>100</td><td>100</td><td>208,131,003</td><td> 1469 ms</td><td>  47 ms</td><td> 32.26x</td>
  </tr>
</table>  
___

NOTE: All CUDA GPU times include all device memsets, host-device memory copies and device-host memory copies.

CPU= Intel i-7 3770K 3.5 Ghz with 3.9 Ghz target

GPU= Tesla K20c 5GB

Windows 7 Ultimate x64

Visual Studio 2010 x64

Would love to see a faster Python version, since that is the *best* language these days. Please contact me with the running time for the same sample sizes!

Python en Ruby zijn talen voor de lui en traag!  

Python und Ruby sind Sprachen für die faul und langsam!  

Python et Ruby sont des langues pour les paresseux et lent!  


<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)
[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)
