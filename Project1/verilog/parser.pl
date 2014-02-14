#!/usr/bin/perl
use strict;
use warnings;
use Verilog::Netlist;

# prepare netlist
my $nl = new Verilog::Netlist();
if ($ARGV[0]) {
    `cat $ARGV[0] modules.v | sed 's/</\[/g' | sed 's/>/\]/g' > top.v`;
}
$nl->read_file(filename => './top.v');
# read in any sub modules
$nl->link();
$nl->lint();
$nl->exit_if_error();

my $scale = 1;
my $netNumber = 0; 
my %netRename = ();
my %netCount = ();
my $cell_count = 1; 
my $wire_count = 1;  
my $port_count = 1; 
my $x; 
my $y;
$netRename{"vdd"} = 0; 
$netRename{"reset"} = 0;
$netRename{"clk"} = 0;
$netRename{"gnd"} = 0; 
for my $mod ( $nl->modules() ) {
    if ( defined($mod->name()) and $ARGV[1] eq $mod->name()) {
        show_hier($mod, '', '');   
    }
}
print $x . " " . $y . " " . $wire_count . " " . $cell_count . " ". $port_count . "\n";
#for my $values ( keys %netCount ) {
#    if ($netCount{$values} == 1) {
#        print "Key: " . $netCount{$values} . " Value: " . $values . "\n";
#    }
#}

sub get_net {
    my $net_t = shift; 
    if (!defined($netRename{$net_t})) {
        $wire_count += 1; 
        $netNumber += 1; 
        $netCount{$net_t} = 1; 
        $netRename{$net_t} = $netNumber; 
        return $netNumber
    }
    $netCount{$net_t} += 1;
    return $netRename{$net_t}; 
}
sub is_output {
    my $pin_t = shift; 
    if ($pin_t eq "Y") {
        return 1;
    }
    if ($pin_t eq "Q") {
        return 1;
    }
    if ($pin_t eq "Z") {
        return 1;
    } 
    if ($pin_t eq "QB") {
        return 1; 
    } 
    return 0;
}
sub show_hier {
   my $mod      = shift;
   my $indent   = shift;
   my $hier     = shift;
   my $size = $mod->cells();
   $y = int(sqrt($size))+1; 
   $x = $y*2;
   my $p_yi = 1; 
   my $p_yo = 1;
   my $p_x = 0;
   for my $sig ($mod->ports) {
        my $portS = 1;
        if (get_net($sig->name) == 0) {
            next;
        }
        if (defined($sig->net()->width)) {
            $portS = scalar $sig->net()->width();
            for (my $i = 0; $i < $portS; $i++) {                
                $port_count += 1; 
                print 'p name=',$sig->name(),'_',$i,' wire=', get_net($sig->name() . '[' . $i . ']');
                $p_x = 0;
                if ($sig->direction() eq "out") {
                    $p_x = $x;
                    $p_yo += $scale;
                    print ' y=',$p_yo;
                } else {
                    $p_yi +=$scale;
                    print ' y=',$p_yi;
                }
                print ' x=',$p_x,"\n";
            }
        } else {
            $port_count += 1; 
            $p_x = 0; 
            print 'p name=',$sig->name(),' wire=',get_net($sig->name());
            $p_x = 0;
            if ($sig->direction() eq "out") {
                $p_x = $x;
                $p_yo += $scale;
                print ' y=',$p_yo;
            } else {
                    $p_yi +=$scale;
                    print ' y=',$p_yi;
            }
            print ' x=',$p_x,"\n";
        } 
    }
   if ($p_yi > $y) {
      $y = $p_yi;
    }
    if ($p_yo > $y) {
      $y = $p_yo;
    } 

   for my $cell ($mod->cells_sorted()) {
       $cell_count += 1;
       print 'g name=',$cell->name;
       for my $pin ($cell->pins()) {
            my $net_t = get_net($pin->netname());
            if ($net_t != 0) {
                if (is_output($pin->name)) {
                    print " fanout=" . $net_t;
                } else {
                    print " fanin=" . $net_t;
                }
            }
        }
        print "\n";
   }
}
sub sigdir {
   # Change "in"  to "input"
   # Change "out" to "output"
   my $dir = shift;
   return ($dir eq 'inout') ? $dir : $dir . 'put';
}
