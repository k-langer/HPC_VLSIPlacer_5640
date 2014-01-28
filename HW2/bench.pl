#!/usr/bin/perl
use strict;
use warnings;

print ",sequential,parallel,reduce\n";
for (my $i=1; $i <= 1000000000; $i *= 10) {
    my $sa = `./HW2 $i s`;
    my $pa = `./HW2 $i p`;
    my $ra = `./HW2 $i r`;    
    print "$i, ";
    if ($sa =~ /total\stime\s\(ms\): (\d+\.\d+)/) {
        print "$1, ";
    }
    if ($pa =~ /total\stime\s\(ms\): (\d+\.\d+)/) {
        print "$1, ";
    }
    if ($ra =~ /total\stime\s\(ms\):\s(\d+\.\d+)/) {
        print "$1\n";
    }
}
