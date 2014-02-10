use strict;
use warnings;
use Verilog::Netlist;

# prepare netlist
my $nl = new Verilog::Netlist();
$nl->read_file(filename => './top.v');

# read in any sub modules
$nl->link();
$nl->lint();
$nl->exit_if_error();

for my $mod ( $nl->top_modules_sorted() ) {
   show_hier($mod, '', '');
}

my $netNumber = 1; 
my %netRename = ();
sub get_net {
    my $net_t = shift; 
    if (!defined($netRename{$net_t})) {
        $netNumber += 1; 
    }
    $netRename{$net_t} = $netNumber; 
    return $netNumber; 
}
sub show_hier {
   my $mod      = shift;
   my $indent   = shift;
   my $hier     = shift;
   my $size = $mod->cells();
   my $y = int(sqrt($size))+1; 
   my $x = $y*2;
   my $p_y = 1; 
   my $p_x = 0; 
   for my $sig ($mod->ports) {
        my $portS = 1;
        if (defined($sig->net()->width)) {
            $portS = scalar $sig->net()->width();
            for (my $i = 0; $i < $portS; $i++) {
                $p_y += 1;
                
                print 'p name=',$sig->name(),'_',$i,' wire=', get_net($sig->name() . '[' . $i . ']') ,' y=',$p_y;
                $p_x = 0;
                if ($sig->direction() eq "out") {
                    $p_x = $x;
                }
                print ' x=',$p_x,"\n";
            }
        } else {
            $p_y += 1;
            $p_x = 0; 
            print 'p name=',$sig->name(),' wire=',$sig->name(),' y=',$p_y;
            $p_x = 0;
            if ($sig->direction() eq "out") {
                $p_x = $x;
            }
            print ' x=',$p_x,"\n";
        } 
    }

   for my $cell ($mod->cells_sorted()) {
       #print 'Cell= ',$cell->name,"\n";
       #for my $pin ($cell->pins_sorted()) {
       #        print $indent, ' PinName=', $pin->name(), ' NetName=', $pin->netname(), "\n";
      #}
   }
}
sub sigdir {
   # Change "in"  to "input"
   # Change "out" to "output"
   my $dir = shift;
   return ($dir eq 'inout') ? $dir : $dir . 'put';
}
